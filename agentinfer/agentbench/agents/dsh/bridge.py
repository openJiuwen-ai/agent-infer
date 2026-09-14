# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

"""Task-local JavaScript bridge between AgentBench and DeepSeek Harness."""

AGENTBENCH_BRIDGE = """\
import { AsyncLocalStorage } from 'node:async_hooks'

export const inject = ['llm', 'planMode', 'userQuestions']

const lineageStorage = new AsyncLocalStorage()
const fetchWrapperState = Symbol('agentbench-bridge.fetchWrapperState')

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

function isChatCompletionsRequest(request) {
  return request.method === 'POST' && new URL(request.url).pathname.endsWith('/chat/completions')
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
  const originalFetch = globalThis.fetch

  let installedFetch
  if (typeof rootSessionId !== 'string' || !rootSessionId) {
    ctx.logger?.warn?.('agentbench-bridge: AGENTBENCH_ROOT_SESSION_ID unavailable; requests have no canonical lineage')
  } else {
    const previousFetch = globalThis.fetch
    installedFetch = async (input, init) => {
      const lineage = lineageStorage.getStore()
      if (lineage === undefined) return activeFetch(previousFetch)(input, init)
      const request = new Request(input, init)
      if (!isChatCompletionsRequest(request)) return activeFetch(previousFetch)(request)
      return activeFetch(previousFetch)(await encodeRequest(request, lineage))
    }
    installedFetch[fetchWrapperState] = { previous: previousFetch, disposed: false }
    globalThis.fetch = installedFetch
  }

  const disposeAgentCreated = ctx.on('agent/created', ({ agent }) => {
    const lineage = sessionLineage(agent.session.header, rootSessionId)
    if (lineage !== undefined) lineages.set(lineage.session_id, lineage)
    if (!forcePlanMode || lineage?.agent_role !== 'lead') return
    // Fail closed: a synchronous throw here vetoes agent publication, so the
    // run exits nonzero instead of degrading to prompt-only planning.
    const planMode = ctx.get('planMode')
    if (planMode === undefined) {
      throw new Error('agentbench-bridge: planMode service unavailable; cannot enforce plan mode')
    }
    planMode.set(agent, true)
  })

  const disposeStream = ctx.on('llm/stream', async function* (options, next) {
    const lineage = lineages.get(String(options.sessionId))
    const stream = next()
    if (lineage === undefined) {
      yield* stream
      return
    }
    while (true) {
      const item = await lineageStorage.run(lineage, () => stream.next())
      if (item.done) return
      yield item.value
    }
  })

  if (autoApprove) {
    // Fail closed with the same veto: without an answerer, exit_plan_mode
    // would hang the headless run instead of completing the plan review.
    const userQuestions = ctx.get('userQuestions')
    if (userQuestions === undefined) {
      throw new Error('agentbench-bridge: userQuestions service unavailable; cannot auto-approve the plan review')
    }
    userQuestions.registerProvider({
      ask: async (request) => ({
        answers: request.questions.map((q) => {
          if (q.intent?.kind === 'plan-review') {
            return { id: q.id, selected: [q.intent.approve] }
          }
          const [defaultOption] = q.options
          if (defaultOption === undefined) {
            throw new Error('agentbench-bridge cannot answer a question without options')
          }
          return { id: q.id, selected: [defaultOption.value] }
        }),
      }),
    })
  }

  return () => {
    disposeStream()
    disposeAgentCreated()
    if (installedFetch !== undefined) {
      installedFetch[fetchWrapperState].disposed = true
      if (globalThis.fetch === installedFetch) globalThis.fetch = activeFetch(originalFetch)
    }
  }
}
"""
