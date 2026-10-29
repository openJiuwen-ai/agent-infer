# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

"""Task-local JavaScript bridge between AgentBench and DeepSeek Harness."""

AGENTBENCH_BRIDGE = """\
import { AsyncLocalStorage } from 'node:async_hooks'
import { randomUUID } from 'node:crypto'
import { writeFileSync } from 'node:fs'

export const inject = ['llm', 'planMode', 'userQuestions', 'tools', 'systemPrompt', 'subagents']

const lineageStorage = new AsyncLocalStorage()
const fetchWrapperState = Symbol('agentbench-bridge.fetchWrapperState')
const SUBAGENT_STEP_LIMIT = 100
const AGENT_STEP_LIMIT = 200
const TOOL_WINDOW = 8
const REPEAT_LIMIT = 4

function activeFetch(fetch) {
  const state = fetch?.[fetchWrapperState]
  return state?.disposed ? activeFetch(state.previous) : fetch
}

function sessionLineage(header, rootSessionId) {
  const sessionId = header?.id
  if (typeof sessionId !== 'string' || !sessionId) return undefined
  const parentSessionId = header.parentSession
  // Loose null check keeps this aligned with session_is_root in transcript.py,
  // which treats both absent and JSON-null parentSession as root.
  const isRoot = parentSessionId == null && (header.delegationDepth ?? 0) === 0
  if ((!isRoot || header.origin === 'subagent') && !parentSessionId) {
    throw new Error('agentbench-bridge: child session is missing parentSession')
  }
  return {
    program_id: sessionId,
    task_id: rootSessionId,
    session_id: sessionId,
    agent_id: sessionId,
    parent_program_id: isRoot ? undefined : parentSessionId,
    blocks_parent: !isRoot,
    expected_resume: isRoot,
    agent_role: isRoot ? 'lead' : 'subagent',
  }
}

function isChatCompletionsRequest(request, apiBaseURL) {
  const url = new URL(request.url)
  return request.method === 'POST' && url.origin === new URL(apiBaseURL).origin
    && url.pathname === '/v1/chat/completions'
}

function encodeRequest(request, lineage) {
  return request.clone().text().then((body) => {
    const payload = JSON.parse(body)
    if (payload === null || typeof payload !== 'object' || Array.isArray(payload)) return request
    const xargs = payload.vllm_xargs
    if (xargs !== undefined && (xargs === null || typeof xargs !== 'object' || Array.isArray(xargs))) {
      throw new TypeError('DSH request vllm_xargs must be an object')
    }
    payload.vllm_xargs = {
      ...xargs,
      agentic_context: JSON.stringify(
        Object.fromEntries(Object.entries(lineage).filter(([, value]) => value !== undefined)),
      ),
    }
    const headers = new Headers(request.headers)
    headers.delete('content-length')
    return new Request(request, { body: JSON.stringify(payload), headers })
  })
}

export function apply(ctx, config = {}) {
  const { forcePlanMode = false, autoApprove = false } = config
  const rootSessionId = process.env.AGENTBENCH_ROOT_SESSION_ID
  const lineages = new Map()
  const subagentOutcomes = new Map()
  const budgets = new Map()
  let autoPlanApprovals = 0
  let planRecoveries = 0
  let failure
  let unansweredQuestion
  const persistLineage = () => {
    if (config.lineagePath) writeFileSync(config.lineagePath, JSON.stringify({
      task_id: rootSessionId,
      sessions: [...lineages.values()],
      auto_plan_approvals: autoPlanApprovals,
      plan_recoveries: planRecoveries,
      failure,
      unanswered_question: unansweredQuestion,
      subagent_outcomes: Object.fromEntries(subagentOutcomes),
    }, null, 2) + '\\n')
  }
  const recordFailure = (reason, sessionId, message) => {
    failure ??= { reason, session_id: sessionId, message }
    persistLineage()
    return failure.message
  }
  const originalFetch = globalThis.fetch

  let installedFetch
  if (typeof rootSessionId !== 'string' || !rootSessionId) {
    ctx.logger?.warn?.('agentbench-bridge: AGENTBENCH_ROOT_SESSION_ID unavailable; requests have no canonical lineage')
  } else {
    const previousFetch = globalThis.fetch
    installedFetch = async (input, init) => {
      const request = new Request(input, init)
      if (!isChatCompletionsRequest(request, config.apiBaseURL)) return activeFetch(previousFetch)(request)
      const lineage = lineageStorage.getStore()
      if (lineage === undefined) throw new Error('agentbench-bridge: model request has no session lineage')
      return activeFetch(previousFetch)(await encodeRequest(request, lineage))
    }
    installedFetch[fetchWrapperState] = { previous: previousFetch, disposed: false }
    globalThis.fetch = installedFetch
  }

  const disposeAgentCreated = ctx.on('agent/created', ({ agent }) => {
    const lineage = sessionLineage(agent.session.header, rootSessionId)
    if (lineage !== undefined) {
      lineages.set(lineage.session_id, lineage)
      budgets.set(lineage.session_id, { steps: 0, reads: [], errors: [] })
    }
    persistLineage()
    if (lineage?.agent_role === 'subagent') {
      const budget = budgets.get(lineage.session_id)
      agent.ctx.tools.restrict({ allow: ['glob', 'grep', 'read'] })
      agent.ctx.tools.guard(({ name, arguments: args }) => {
        const key = name === 'read' && args !== null && typeof args === 'object'
          ? JSON.stringify([args.file_path, args.offset ?? 1, args.limit ?? 2000]) : null
        budget.reads.push(key)
        if (budget.reads.length > TOOL_WINDOW) budget.reads.shift()
        if (key !== null && budget.reads.filter(value => value === key).length >= REPEAT_LIMIT) {
          return recordFailure('subagent_read_loop', lineage.session_id,
            `Explore subagent repeated the same read ${REPEAT_LIMIT} times within ${TOOL_WINDOW} tool calls`)
        }
      })
      agent.ctx.systemPrompt.section({
        name: 'agentbench-explore', order: 0,
        text: `You are a read-only Explore subagent. Search first, read focused ranges with limit <= 200, then report concise evidence to the lead. Do not implement fixes or delegate further. You have at most ${SUBAGENT_STEP_LIMIT} steps; finish with your findings.`,
      })
    }
    if (!forcePlanMode || lineage?.agent_role !== 'lead') return
    // Fail closed: a synchronous throw here vetoes agent publication, so the
    // run exits nonzero instead of degrading to prompt-only planning.
    const planMode = ctx.get('planMode')
    if (planMode === undefined) {
      throw new Error('agentbench-bridge: planMode service unavailable; cannot enforce plan mode')
    }
    planMode.set(agent, true)
    agent.ctx.tools.guard(({ name }) => {
      if (!planMode.get(agent).active) {
        if (name === 'exit_plan_mode') {
          return 'Plan mode is already inactive. Do not call exit_plan_mode again; continue implementation or finish with a final response and no tool call.'
        }
        return
      }
      const children = [...lineages.values()].filter(row => row.parent_program_id === agent.id)
      if (children.length === 0 && name !== 'subagent') {
        return 'Delegate repository exploration with subagent before using other tools.'
      }
      if (name === 'exit_plan_mode' && children.some(row => subagentOutcomes.get(row.session_id) !== 'completed')) {
        return 'Wait for the Explore subagent report before presenting the plan. You may end this turn to await its completion notice.'
      }
      if (!['subagent', 'exit_plan_mode', 'read', 'glob', 'grep', 'working_directory'].includes(name)) {
        return 'Plan mode is read-only. Wait for exploration, then call exit_plan_mode before implementation.'
      }
    })
  })

  const disposePreStep = ctx.on('agent/pre-step', async ({ agent }, next) => {
    if (failure) throw new Error(failure.message)
    const budget = budgets.get(agent.id)
    const child = lineages.get(agent.id)?.agent_role === 'subagent'
    const limit = child ? SUBAGENT_STEP_LIMIT : AGENT_STEP_LIMIT
    if (budget && ++budget.steps > limit) {
      throw new Error(recordFailure(child ? 'subagent_step_limit' : 'agent_step_limit', agent.id,
        `Agent exceeded ${limit} steps`))
    }
    return next()
  })

  const disposeToolResult = ctx.on('tools/result', (exec, result) => {
    const budget = budgets.get(exec.agent?.id)
    if (!budget) return
    const invalid = result.isError && (result.error.info?.code === 'INVALID_ARGS'
      || (exec.name === 'exit_plan_mode' && !ctx.get('planMode')?.get(exec.agent).active))
    const key = invalid ? JSON.stringify([exec.name, result.error.message]) : null
    budget.errors.push(key)
    if (budget.errors.length > TOOL_WINDOW) budget.errors.shift()
    if (key !== null && budget.errors.filter(value => value === key).length >= REPEAT_LIMIT) {
      recordFailure('validation_error_loop', exec.agent.id,
        `Repeated invalid tool call ${exec.name}: ${result.error.message}`)
    }
  })

  const disposeStopping = ctx.on('agent/turn-stopping', ({ agent }) => {
    if (!forcePlanMode || lineages.get(agent.id)?.agent_role !== 'lead') return
    if (failure) throw new Error(failure.message)
    if (!ctx.get('planMode').get(agent).active) return
    const children = [...lineages.values()].filter(row => row.parent_program_id === agent.id)
    if (children.some(row => !subagentOutcomes.has(row.session_id))) return
    if (planRecoveries > 0) {
      throw new Error(recordFailure('plan_not_approved', agent.id, 'Lead ended without an approved plan after one recovery reminder'))
    }
    planRecoveries++
    persistLineage()
    agent.inject({
      id: randomUUID(), role: 'user', source: { kind: 'agentbench-plan-recovery' },
      content: [{ type: 'text', text: children.length === 0
        ? 'Plan mode is still active. Delegate read-only exploration with subagent, then submit your plan through exit_plan_mode before implementing.'
        : 'All exploration reports are available. Plan mode is still active. Call exit_plan_mode with your plan, then implement the fix after approval. Do not finish with analysis alone.' }],
    })
  })

  const disposeSubagentEnd = ctx.on('subagent/end', ({ id, stopReason }) => {
    subagentOutcomes.set(id, stopReason)
    if (stopReason !== 'completed') recordFailure('subagent_failed', id, `Explore subagent ended with ${stopReason}`)
    else persistLineage()
  })

  const disposeStream = ctx.on('llm/stream', async function* (options, next) {
    if (unansweredQuestion) throw new Error(unansweredQuestion)
    const lineage = lineages.get(String(options.sessionId))
    if (lineage === undefined) throw new Error('agentbench-bridge: unknown model session ' + options.sessionId)
    const stream = next()
    try {
      while (true) {
        const item = await lineageStorage.run(lineage, () => stream.next())
        if (item.done) return
        yield item.value
      }
    } finally {
      if (stream.return) await lineageStorage.run(lineage, () => stream.return())
    }
  })

  if (autoApprove) {
    // Fail closed with the same veto: without an answerer, exit_plan_mode
    // would hang the headless run instead of completing the plan review.
    const userQuestions = ctx.get('userQuestions')
    if (userQuestions === undefined) {
      throw new Error('agentbench-bridge: userQuestions service unavailable; cannot auto-approve the plan review')
    }
  }
  const disposeQuestions = ctx.on('user-questions/request', async (request) => ({
        answers: request.questions.map((q) => {
          if (autoApprove && q.intent?.kind === 'plan-review') {
            autoPlanApprovals++
            persistLineage()
            return { id: q.id, selected: [q.intent.approve] }
          }
          unansweredQuestion = 'agentbench-bridge: unanswered user question; only plan-review is supported'
          persistLineage()
          throw new Error(unansweredQuestion)
        }),
      }))

  return () => {
    disposeStream()
    disposeAgentCreated()
    disposePreStep()
    disposeToolResult()
    disposeStopping()
    disposeSubagentEnd()
    disposeQuestions?.()
    if (installedFetch !== undefined) {
      installedFetch[fetchWrapperState].disposed = true
      if (globalThis.fetch === installedFetch) globalThis.fetch = activeFetch(originalFetch)
    }
  }
}
"""
