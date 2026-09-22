// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project
/**
 * Build the Chinese RSI guide with @oai/artifact-tool (ES modules).
 *
 * Prerequisites: @oai/artifact-tool, playwright, and the installed presentations
 * skill with its finalizer. Set ARTIFACT_TOOL_MODULES when packages are supplied
 * outside node's normal resolution path. This script never runs model services.
 *
 * node tools/rsi/build_presentation.mjs --build-dir /outside/repo/ppt-build \
 *   --skill-dir /path/to/presentations/skill --python /path/to/python \
 *   --dashboard /path/to/dashboard.html
 *
 * Temporary renders and validation receipts remain in --build-dir. The final
 * deck and editable diagrams are copied to docs/assets/rsi after validation.
 */
import fs from 'node:fs/promises';
import path from 'node:path';
import { createRequire } from 'node:module';
import { fileURLToPath, pathToFileURL } from 'node:url';

const arg = (name, fallback) => {
  const index = process.argv.indexOf(`--${name}`);
  return index < 0 ? fallback : process.argv[index + 1];
};
const repo = path.resolve(arg('repo', path.join(path.dirname(fileURLToPath(import.meta.url)), '../..')));
const buildDir = path.resolve(arg('build-dir', path.join(repo, '../rsi-ppt-build')));
const skillDir = arg('skill-dir', process.env.PRESENTATIONS_SKILL_DIR);
const python = arg('python', process.env.RUNTIME_PYTHON || 'python');
const dashboardPath = arg('dashboard', path.join(repo, 'agentinfer/rsi/dashboard/static/dashboard.html'));
const font = arg('font', 'Microsoft YaHei');
const modules = process.env.ARTIFACT_TOOL_MODULES;
if (modules && !process.env.RUNTIME_NODE_MODULES) process.env.RUNTIME_NODE_MODULES = modules;
if (!skillDir || !path.isAbsolute(skillDir)) throw new Error('--skill-dir must be an absolute presentations skill path');
if (buildDir === repo || buildDir.startsWith(repo + path.sep)) throw new Error('--build-dir must be outside the repository');
const requirePackage = modules ? createRequire(path.join(modules, '_rsi_loader.cjs')) : createRequire(import.meta.url);
const importPackage = async (name) => import(pathToFileURL(requirePackage.resolve(name)).href);
const { Presentation, PresentationFile, FileBlob } = await importPackage('@oai/artifact-tool');
const { finalizePresentation, resolvePresentationFont } = await import(pathToFileURL(path.join(skillDir, 'container_tools/artifact_tool_utils.mjs')).href);
const family = resolvePresentationFont({ fontFamily: font });
const outDir = path.join(repo, 'docs/assets/rsi');
await fs.mkdir(buildDir, { recursive: true });
await fs.mkdir(outDir, { recursive: true });

const C = { bg:'#F3F4EE', white:'#FFFEFA', ink:'#203128', muted:'#68776A', green:'#347151', lime:'#D8EC95', soft:'#E8F0E2', line:'#CDD8C8', amber:'#A46525', amberBg:'#FAF0DF', red:'#A95040', blue:'#DCEAF0' };
const W=1600, H=900;
const sources = {
  blog:'https://z.ai/blog/glm-built-its-inference-infrastructure',
  vllm:'https://docs.vllm.ai/en/latest/design/arch_overview.html',
  profiling:'https://docs.vllm.ai/en/latest/contributing/profiling/',
  ascend:'https://docs.vllm.ai/projects/ascend/en/main/developer_guide/performance_and_debug/service_profiling_guide.html',
  agentinfer:'https://github.com/openJiuwen-ai/agent-infer/blob/main/docs/zh/explanation/architecture.md',
  replay:'https://github.com/openJiuwen-ai/agent-infer/blob/main/docs/zh/how-to/run-trace-replay.md',
  benchmark:'https://github.com/openJiuwen-ai/agent-infer/blob/main/docs/zh/explanation/benchmark-methodology.md',
  zcode:'https://github.com/zai-org/ZCode',
  zcodeCli:'https://github.com/zai-org/ZCode/blob/main/apps/zcode-cli/packages/cli/src/prompt-command.ts',
  model:'https://huggingface.co/zai-org/GLM-5.3-Flash',
};
const presentation=Presentation.create({slideSize:{width:W,height:H}});
const slides=[];
const shapesBySlide=new Map();
const svgXml=(s)=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&apos;'}[c]));
const shape=(s,x,y,w,h,{fill=C.white,line=C.line,width=1.5,geometry='rect',name=''}={})=>s.shapes.add({geometry,name,position:{left:x,top:y,width:w,height:h},fill,line:{fill:line,width}});
function text(s,x,y,w,h,value,size=28,color=C.ink,bold=false,align='left') {
  const sh=shape(s,x,y,w,h,{fill:'none',line:'none',width:0,geometry:'textbox',name:value.slice(0,35)});
  sh.text=value;
  sh.text.style={typeface:family,fontSize:size,color,bold,alignment:align,verticalAlignment:'middle',autoFit:'none',wrap:'square',insets:{left:0,right:0,top:0,bottom:0}};
  return sh;
}
function rule(s,x,y,w,color=C.line){return shape(s,x,y,w,1.5,{fill:color,line:'none',width:0});}
function box(s,x,y,w,h,title,subtitle='',fill=C.white,titleSize=25) {
  const sh=shape(s,x,y,w,h,{fill});
  text(s,x+20,y+12,w-40,subtitle?36:h-24,title,titleSize,C.ink,true);
  if(subtitle)text(s,x+20,y+52,w-40,h-64,subtitle,21,C.muted);
  return sh;
}
function connect(s,a,b,from='right',to='left',dashed=false,color=C.green){return s.shapes.connect(a,b,{kind:'elbow',fromSide:from,toSide:to,line:{fill:color,width:dashed?1.8:2.2,style:dashed?'dashed':'solid'},tail:{type:'triangle',width:'sm',length:'sm'}});}
function routed(s,a,b,points,from,to,color=C.green) {
  const opposite={right:'left',left:'right',bottom:'top',top:'bottom'};
  const anchors=points.map(([x,y])=>shape(s,x-.5,y-.5,1,1,{fill:'none',line:'none',width:0}));
  const route=[a,...anchors,b];
  for(let k=0;k<route.length-1;k++) {
    let direction=k===0?from:opposite[to];
    if(k>0&&k<route.length-2){const dx=points[k][0]-points[k-1][0],dy=points[k][1]-points[k-1][1];direction=Math.abs(dx)>Math.abs(dy)?(dx>=0?'right':'left'):(dy>=0?'bottom':'top');}
    s.shapes.connect(route[k],route[k+1],{kind:'straight',fromSide:k===0?from:direction,toSide:k===route.length-2?to:opposite[direction],line:{fill:color,width:2,style:'dashed'},tail:{type:k===route.length-2?'triangle':'none',width:'sm',length:'sm'}});
  }
}
function notes(s,content,urls=[]){s.speakerNotes.textFrame.setText(content+'\n\n来源与边界：\n'+urls.map(u=>'• '+u).join('\n')+'\n本稿中的目标架构为设计建议。性能数字若出现均为演示，不代表实测。GLM-5.3-Flash 权重保持固定。');}
function slide(title,subtitle='',narrative='',urls=[]) {
  const s=presentation.slides.add();slides.push(s);s.background.fill=C.bg;
  text(s,64,42,1472,70,title,44,C.ink,true);
  if(subtitle)text(s,66,114,1468,46,subtitle,24,C.muted);
  rule(s,64,174,1472);
  text(s,64,846,1300,25,'AgentInfer RSI  /  离线脚手架与目标架构  /  2026-09-21',15,C.muted);
  text(s,1470,843,66,30,String(slides.length).padStart(2,'0'),18,C.green,false,'right');
  notes(s,narrative,urls);return s;
}
function statement(s,x,y,w,label,body){text(s,x,y,w,46,label,31,C.green,true);text(s,x,y+58,w,130,body,27,C.ink);}

// Engineering diagrams have two outputs from the same editable specification:
// native PowerPoint objects, and editable SVG. Mermaid files carry logical edges.
function drawDiagram(s, spec, {x=0,y=0,scale=1}={}) {
  const nodes=new Map();
  for(const n of spec.nodes) {
    const sh=shape(s,x+n.x*scale,y+n.y*scale,n.w*scale,n.h*scale,{fill:n.fill||C.white,line:n.line||C.line,width:1.5});nodes.set(n.id,sh);
    const lines=n.lines||[n.title];
    const mainSize=(n.size||25)*scale;
    if(lines.length===1)text(s,x+(n.x+16)*scale,y+(n.y+6)*scale,(n.w-32)*scale,(n.h-12)*scale,lines[0],mainSize,C.ink,true,n.align||'left');
    else {
      text(s,x+(n.x+16)*scale,y+(n.y+10)*scale,(n.w-32)*scale,36*scale,lines[0],mainSize,C.ink,true);
      text(s,x+(n.x+16)*scale,y+(n.y+50)*scale,(n.w-32)*scale,(n.h-60)*scale,lines.slice(1).join('\n'),(n.bodySize||21)*scale,C.muted);
    }
  }
  for(const e of spec.edges||[]) {
    if(!e.via){connect(s,nodes.get(e.a),nodes.get(e.b),e.from||'right',e.to||'left',!!e.dashed,e.color||C.green);continue;}
    const anchors=e.via.map(([px,py])=>shape(s,x+px*scale-.5,y+py*scale-.5,1,1,{fill:'none',line:'none',width:0}));
    const route=[nodes.get(e.a),...anchors,nodes.get(e.b)];
    const endpoint=(id,side)=>{const n=spec.nodes.find(v=>v.id===id);return ({left:[n.x,n.y+n.h/2],right:[n.x+n.w,n.y+n.h/2],top:[n.x+n.w/2,n.y],bottom:[n.x+n.w/2,n.y+n.h]})[side];};
    const pts=[endpoint(e.a,e.from),...e.via,endpoint(e.b,e.to)];
    for(let k=0;k<route.length-1;k++) {
      const [dx,dy]=[pts[k+1][0]-pts[k][0],pts[k+1][1]-pts[k][1]];
      const direction=Math.abs(dx)>Math.abs(dy)?(dx>=0?'right':'left'):(dy>=0?'bottom':'top');
      s.shapes.connect(route[k],route[k+1],{kind:'straight',fromSide:k===0?e.from:direction,toSide:k===route.length-2?e.to:({right:'left',left:'right',bottom:'top',top:'bottom'})[direction],line:{fill:e.color||C.green,width:1.8,style:'dashed'},tail:{type:k===route.length-2?'triangle':'none',width:'sm',length:'sm'}});
    }
  }
  for(const l of spec.labels||[])text(s,x+l.x*scale,y+l.y*scale,l.w*scale,l.h*scale,l.text,(l.size||22)*scale,l.color||C.muted,l.bold||false,l.align||'left');
  shapesBySlide.set(s,spec);
}
function svgDiagram(spec,title) {
  const edgePath=(e)=>{
    const a=spec.nodes.find(n=>n.id===e.a),b=spec.nodes.find(n=>n.id===e.b);
    const pt=(n,side)=>({left:[n.x,n.y+n.h/2],right:[n.x+n.w,n.y+n.h/2],top:[n.x+n.w/2,n.y],bottom:[n.x+n.w/2,n.y+n.h]})[side];
    const from=e.from||'right',to=e.to||'left';const [x1,y1]=pt(a,from),[x2,y2]=pt(b,to);
    if(e.via)return `M${x1},${y1} `+[...e.via,[x2,y2]].map(([px,py])=>`L${px},${py}`).join(' ');
    if(from===to&&(from==='left'||from==='right')){const bend=from==='left'?Math.min(x1,x2)-24:Math.max(x1,x2)+24;return `M${x1},${y1} H${bend} V${y2} H${x2}`;}
    return (from==='top'||from==='bottom')?`M${x1},${y1} V${(y1+y2)/2} H${x2} V${y2}`:`M${x1},${y1} H${(x1+x2)/2} V${y2} H${x2}`;
  };
  const body=[`<rect width="1600" height="900" fill="${C.bg}"/>`,`<text x="64" y="90" font-size="44" font-weight="700">${svgXml(title)}</text>`];
  for(const e of spec.edges||[])body.push(`<path d="${edgePath(e)}" fill="none" stroke="${e.color||C.green}" stroke-width="2.3" ${e.dashed?'stroke-dasharray="7 6"':''} marker-end="url(#arrow)"/>`);
  for(const n of spec.nodes){body.push(`<g id="${n.id}"><rect x="${n.x}" y="${n.y}" width="${n.w}" height="${n.h}" fill="${n.fill||C.white}" stroke="${n.line||C.line}" stroke-width="1.5"/>`);const lines=n.lines||[n.title];for(let i=0;i<lines.length;i++)body.push(`<text x="${n.x+16}" y="${n.y+(i?(n.h<95?62:76)+(i-1)*27:34)}" font-size="${i?(n.bodySize||21):(n.size||25)}" font-weight="${i?400:700}" fill="${i?C.muted:C.ink}">${svgXml(lines[i])}</text>`);body.push('</g>');}
  for(const l of spec.labels||[]){const lines=l.text.split('\n');lines.forEach((line,i)=>body.push(`<text x="${l.x}" y="${l.y+(l.size||22)+i*(l.size||22)*1.3}" font-size="${l.size||22}" fill="${l.color||C.muted}" font-weight="${l.bold?700:400}">${svgXml(line)}</text>`));}
  return `<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="900" viewBox="0 0 1600 900"><title>${svgXml(title)}</title><defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8" fill="${C.green}"/></marker></defs><g font-family="Microsoft YaHei, Noto Sans CJK SC, sans-serif" fill="${C.ink}">${body.join('')}</g></svg>`;
}

// 01
{
  const s=slide('AgentInfer RSI 系统设计','固定模型权重下的推理基础设施与 Harness 持续改进',
    '这份讲解稿介绍目标架构和本次初始实现。RSI 在这里指通过可证伪假设、隔离实验与独立验收，持续改进推理系统和 Agent Harness。模型权重保持固定。本次代码只实现可在 CPU 上运行的离线脚手架、持久化演示 Controller、显式反馈评估和本地 Dashboard。两条 GPU/NPU 后端都未进行环境验证。',[sources.blog,sources.model]);
  text(s,74,232,1390,130,'改进对象是系统，\n验收依据是可追溯证据',60,C.ink,true);
  text(s,78,428,1270,105,'控制面管理预算与版本。Agent 提出并实现候选。\n独立裁决决定是否晋升，稳定系统始终可恢复。',31,C.muted);
  rule(s,78,603,1380);
  text(s,78,642,1410,78,'本轮交付：offline scaffold、mock Dashboard、接口契约与设计说明',28,C.green,true);
  text(s,78,733,1410,48,'真实 ZCode、AgentBench、profiler 与部署适配属于后续接入',26,C.muted);
}
// 02
{
  const s=slide('设计依据与工程约束','把模型的工程能力放进可检查的实验流程',
    '智谱文章为这套设计提供了以 AI 参与推理基础设施研发的启发。图中控制面、门禁和知识晋升是本方案的工程设计，不是对文章中系统的逐项复制。固定模型权重、预算、负载与评分器，让一次改动可以被比较。局部反馈要求便宜、及时、客观可验证，最终仍需集成与端到端验收。我们不使用 Agent 自评作为发布依据。',[sources.blog,sources.benchmark]);
  statement(s,78,225,675,'可证伪假设','每个候选说明预期机制、可观察变化，以及何种结果会否定它。');
  statement(s,830,225,680,'局部可验证反馈','参考实现、差分与受控干预缩短定位时间，减少盲目全量试跑。');
  statement(s,78,490,675,'稳定的比较条件','固定模型、负载、硬件、种子与版本。更改条件需要新的实验合同。');
  statement(s,830,490,680,'独立晋升与恢复','验收器、发布策略和 last_good 独立于候选，失败路径同样保留证据。');
}
// 03
const architecture={nodes:[
  {id:'applications',x:64,y:202,w:1010,h:78,lines:['应用与 Agent 入口','Coding / 办公 / Science，ZCode 或 openJiuwen runtime'],size:26,bodySize:21},
  {id:'harness',x:64,y:328,w:1010,h:120,lines:['Harness 与多 Agent 协作','Planner / Profiler / Implementer / Reviewer','任务 DAG、会话、上下文、工具权限、结构化产物'],fill:C.soft},
  {id:'inference',x:64,y:496,w:1010,h:120,lines:['AgentInfer 推理服务层','Semantic Router、Router、工作流观测与准入策略','AgentCache 提供工作流感知，vLLM 保有原生调度与物理 KV'],fill:C.blue},
  {id:'cuda',x:64,y:664,w:485,h:103,lines:['vLLM / CUDA backend','固定 GLM-5.3-Flash，环境未验证'],fill:C.white},
  {id:'ascend',x:589,y:664,w:485,h:103,lines:['vLLM-Ascend backend','独立兼容与性能证据，环境未验证'],fill:C.white},
  {id:'controller',x:1137,y:202,w:399,h:130,lines:['RSI Controller','唯一状态迁移入口','预算、租约、版本指针与恢复'],fill:C.lime},
  {id:'lab',x:1137,y:398,w:399,h:138,lines:['隔离实验与独立裁决','AgentBench / 局部探针 / E2E','候选设备与稳定服务隔离'],fill:C.amberBg},
  {id:'knowledge',x:1137,y:602,w:399,h:138,lines:['证据与分层知识库','原始证据、scope、失效条件','已验证知识进入下一轮'],fill:C.soft},
],edges:[{a:'applications',b:'harness',from:'bottom',to:'top'},{a:'harness',b:'inference',from:'bottom',to:'top'},{a:'inference',b:'cuda',from:'bottom',to:'top'},{a:'inference',b:'ascend',from:'bottom',to:'top'},{a:'controller',b:'lab',from:'bottom',to:'top'},{a:'lab',b:'knowledge',from:'bottom',to:'top'},{a:'controller',b:'harness',from:'left',to:'right',dashed:true},{a:'inference',b:'knowledge',from:'right',to:'left',dashed:true},{a:'knowledge',b:'harness',from:'right',to:'left',dashed:true,via:[[1560,671],[1560,188],[40,188],[40,388]]}],labels:[{x:65,y:784,w:1470,h:42,text:'在线路径与 RSI 旁路分离。知识按 scope 注入下一轮 Harness，模型权重保持固定。',size:23}]};
{
  const s=slide('整体分层架构','目标结构。组件能力以已验证版本与适配实现为准',
    '左侧是在线请求路径，右侧是旁路 RSI 控制面。Controller 不位于每个在线请求的关键路径。Harness 负责任务语义和协作，AgentInfer 负责推理请求的放置、准入和缓存观测。展开 AgentInfer 的目标结构时，Agent Router 内有 Semantic Router；Router 组合 Workflow Profiler、Worker Monitor、API Server、Global Scheduler 与 Tokenizer/TokenCache。AgentCache plugin 侧覆盖工作流感知调度、KV 预测与 NPU 传输，原生 vLLM 保有 waiting/running、模型执行与物理 KV 所有权。这些目标组件不代表本次 RSI 全部实现。AgentBench replay 为受控性能比较提供负载，真实 Agent 任务另验收闭环质量。证据按 scope 进入知识库后由下一轮 Harness 检索。稳定模型服务与候选实验要有独立资源边界。两条后端本次均未环境验证。',[sources.agentinfer,sources.zcode,sources.vllm,sources.replay]);
  drawDiagram(s,architecture);
}
// 04
{
  const s=slide('两条后端共享的引擎 taxonomy','统一定位与反馈标签，保留 CUDA / Ascend 的独立实现和证据',
    'taxonomy 用于给反馈和知识标注归属，并非强制源码目录或严格的层层调用关系。api_server 表示协议与输入输出边界，engine 涉及调度、KV 和引擎状态，worker 涉及设备执行与 model runner，model_scripts 表示模型实现。ops.communication 和 ops.compute 分别归属通信与计算算子。parallel 包含 TP、PP、DP、EP、CP、PD 等执行策略，它跨越 engine、worker、model 和 ops，不能画成严格调用层。本次仅提供 taxonomy 与反馈契约，没有验证两条后端。',[sources.vllm,sources.ascend]);
  text(s,90,215,650,42,'vLLM / CUDA',30,C.green,true);text(s,850,215,650,42,'vLLM-Ascend',30,C.green,true);
  const rows=[['api_server','协议 / 输入 / 流式输出'],['engine','调度 / KV 管理 / 生命周期'],['worker','设备执行 / Model Runner'],['model_scripts','模型实现 / 固定权重加载'],['ops.communication','通信 / KV 传输'],['ops.compute','计算算子 / kernel']];
  rows.forEach(([id,label],i)=>{const y=275+i*70;for(const x of [82,842]){shape(s,x,y,674,59,{fill:i%2?C.white:C.soft});text(s,x+17,y+4,300,51,id,24,C.ink,true);text(s,x+321,y+4,331,51,label,22,C.muted);}});
  text(s,89,715,1420,53,'parallel：跨层执行策略（TP / PP / DP / EP / CP / PD 等）',28,C.green,true);
  text(s,89,779,1420,42,'同一分类不等于同一实现。后端 API、通信与算子差异需要独立验证。',24,C.muted);
}
// 05
{
  const s=slide('多 Agent 协作与 Harness','角色提出产物，Controller 决定派发与状态推进',
    '四个角色是声明式配置与提示词，不需要为每个角色建立 Python 类。Planner 提出 ExperimentSpec，Profiler 形成瓶颈报告，Implementer 提供候选补丁，Reviewer 只提交审查建议。ZCode 以外部 CLI 运行自己的会话和工具循环，Python Harness 对接 start/resume/cancel 和事件流，Controller 负责跨任务编排。语义路由选择推理能力，任务 DAG 选择谁做什么，两者不能混合。当前真实 ZCode/openJiuwen 适配尚未接入。',[sources.zcode,sources.zcodeCli]);
  const a=box(s,75,226,340,123,'Profiler','BottleneckReport\n现象、最小复现、证据');
  const b=box(s,445,226,340,123,'Planner','ExperimentSpec\n假设、预测、证伪条件',C.soft);
  const c=box(s,815,226,340,123,'Implementer','CandidateManifest\n补丁、配置、轻量检查');
  const d=box(s,1185,226,340,123,'Reviewer','ReviewVerdict\n风险与归因检查');
  connect(s,a,b);connect(s,b,c);connect(s,c,d);
  const h=box(s,75,432,1450,127,'统一 Harness 边界','RoleTask、独立 session、上下文版本、tool capability、结构化事件、可恢复运行句柄',C.lime,31);
  for(const role of [a,b,c,d])connect(s,h,role,'top','bottom',true);
  text(s,80,618,710,118,'任务 DAG：依赖、join、重试与预算\nSemantic Router：模型 / 能力选择',27,C.ink);
  text(s,846,618,676,118,'传给推理层：program / session / agent\n父子阻塞关系、request_id、expected_resume',25,C.ink);
  text(s,80,774,1450,38,'单一状态机所有者：Controller。适配器报告事实，不直接晋升候选。',26,C.green,true);
}
// 06
{
  const s=slide('十步闭环与失败返回路径','每一步有主责、产物与明确的版本语义',
    '编号必须保持一致：①基线，②诊断，③假设，④实现审阅，⑤实验，⑥证据，⑦裁决，⑧激活恢复，⑨观察，⑩知识归档。PASS 只生成 accepted，人工批准后才切 active，观察成功才更新 last_good。当前离线 Controller 的 FAIL 和 INCONCLUSIVE 重试都保守返回③，重新冻结假设后再执行⑤。目标设计允许合同不变且证据不充分时增量重测直达⑤，这是后续能力。⑨异常经⑧恢复再归档。恢复失败进入 NEEDS_RECOVERY，停止新轮。模型权重保持固定。');
  const names=['① 基线','② 诊断','③ 假设','④ 实现审阅','⑤ 实验','⑥ 证据','⑦ 裁决','⑧ 激活恢复','⑨ 观察','⑩ 知识归档'];
  const sub=['冻结版本 / 负载','BottleneckReport','ExperimentSpec','候选 / ReviewVerdict','隔离 trial','EvidencePacket','PASS / FAIL / 不充分','active / 恢复记录','SoakResult','scope / 失效条件'];
  const nodes=[];for(let i=0;i<10;i++){const row=i<5?0:1,col=i<5?i:9-i;nodes.push(box(s,78+col*292,231+row*230,270,140,names[i],sub[i],i===6?C.lime:C.white,29));}
  for(let i=0;i<9;i++)connect(s,nodes[i],nodes[i+1],i===4?'bottom':i<4?'right':'left',i===4?'top':i<4?'left':'right');
  routed(s,nodes[9],nodes[0],[[40,531],[40,301]],'left','left');
  text(s,55,396,120,34,'下一轮',20,C.green,true);
  routed(s,nodes[6],nodes[2],[[1089,413],[797,413]],'top','bottom',C.amber);
  text(s,899,378,122,30,'重试',20,C.amber,true);
  routed(s,nodes[8],nodes[7],[[505,633],[797,633]],'bottom','bottom',C.red);
  text(s,548,603,260,28,'观察失败 → 恢复',19,C.red,true);
  text(s,84,670,1428,60,'当前脚手架：⑦ FAIL / INCONCLUSIVE 重试均返回③，重新冻结后进入⑤。',26,C.green,true);
  text(s,84,742,1428,72,'目标增量重测可直达⑤；⑨异常经⑧恢复。⑩归档后可开始下一轮①。',25,C.muted);
}
// 07
const feedbackDiagram={nodes:[
  {id:'hypothesis',x:64,y:215,w:300,h:125,lines:['③ 可证伪假设','机制预测 / 否定条件'],fill:C.lime},
  {id:'probe',x:432,y:215,w:635,h:125,lines:['④⑤ 局部探针与观测','参考 / 差分 / 受控干预 / 消融 + profile'],fill:C.white},
  {id:'record',x:1135,y:215,w:401,h:125,lines:['⑥ FeedbackRecord','结果 / oracle / scope / evidence'],fill:C.soft},
  {id:'fix',x:1135,y:435,w:401,h:115,lines:['④ 局部修正','依据客观差异缩小搜索范围'],fill:C.white},
  {id:'integrate',x:610,y:435,w:457,h:115,lines:['⑤ 集成与真实任务 E2E','固定环境下验收质量与性能'],fill:C.white},
  {id:'gate',x:64,y:435,w:478,h:115,lines:['⑦ 独立裁决','通过后进入⑧激活与⑨观察'],fill:C.amberBg},
  {id:'knowledge',x:64,y:673,w:478,h:105,lines:['⑩ 知识晋升 / 反例保留','证据齐全 + scope + 失效条件'],fill:C.soft},
  {id:'retrieve',x:610,y:673,w:926,h:105,lines:['下一轮①基线与③规划','检索 system 契约、当前层知识与相邻依赖；记录引用版本'],fill:C.lime},
],edges:[{a:'hypothesis',b:'probe'},{a:'probe',b:'record'},{a:'record',b:'fix',from:'bottom',to:'top'},{a:'fix',b:'integrate',from:'left',to:'right'},{a:'integrate',b:'gate',from:'left',to:'right'},{a:'gate',b:'knowledge',from:'bottom',to:'top'},{a:'knowledge',b:'retrieve'},{a:'fix',b:'probe',from:'top',to:'bottom',dashed:true},{a:'knowledge',b:'hypothesis',from:'left',to:'left',dashed:true}],labels:[{x:327,y:564,w:270,h:89,text:'经⑧激活 / ⑨观察\n异常先恢复',size:20},{x:610,y:570,w:926,h:83,text:'局部反馈低成本、及时、客观可验证，检查顺序由假设决定。\nprofile 运行不能作为最终性能验收。',size:22}]};
{
  const s=slide('稠密反馈与知识回路','局部迭代缩短反馈距离，集成与 E2E 保留最终约束',
    '稠密反馈的关键是局部、低成本、及时、客观可验证。以可证伪假设发起参考实现、差分、受控干预、消融或 profile。FeedbackRecord 保存 oracle、实际结果、原始证据和 scope。Agent 根据差异修正，再通过集成与真实 Agent 任务验收。正确性、系统行为与性能三类检查的先后由假设决定。带 profiler 的运行仅用于归因，不作最终性能验收。观测或失败也可以形成知识，但必须保留适用范围与限制。',[sources.blog,sources.profiling,sources.ascend]);
  drawDiagram(s,feedbackDiagram);
}
// 08
{
  const s=slide('三类反馈与探针选择','检查顺序由当前假设决定，结果必须有可重复的 oracle',
    '三类反馈各自回答不同问题。正确性比较数值、输出协议或任务结果；系统行为比较资源回收、调度公平性、路由亲和、请求生命周期；性能比较可重复负载下的延迟与吞吐。某个假设可以先验证行为，再看性能，也可以先做差分正确性。不要把类别画成固定流水线。以下是建议的探针模式，不代表本次已连接硬件执行器。',[sources.vllm,sources.profiling,sources.ascend]);
  const rows=[['correctness','参考实现 / 差分 / 容差检查','协议或数值是否满足冻结合同'],['system_behavior','受控干预 / 状态不变量 / 故障注入','行为是否符合预测，资源是否收敛'],['performance','microbench / 配对负载 / 消融','去掉 profiler 后收益是否仍成立']];
  rows.forEach(([a,b,c],i)=>{const y=228+i*176;text(s,78,y,365,50,a,31,C.green,true);text(s,480,y,1015,50,b,29,C.ink,true);text(s,480,y+66,1015,62,c,28,C.muted);if(i<2)rule(s,78,y+145,1445);});
}
// 09
{
  const s=slide('FeedbackRecord 的最小契约','每个结果都能回答：测了谁、在哪测、如何判断、证据在哪',
    '本页列出当前 FeedbackRecord 的主要实际字段，完整输入以 examples/rsi/feedback.json 和 agentinfer/rsi/feedback/models.py 为准。scope 必须包含 engine_version、model_revision 与 workload_id。diagnostic 标记采样可能扰动性能，synthetic 标记演示记录。pass/fail 需要有限测量值，unavailable 必须用 null。evidence_refs 目前是溯源指针，不构成对外部产物的真实性认证。本轮 evaluate 消费显式 JSON records 并输出报告，不会执行 GPU/NPU profiler。');
  const a=[['关联','check_id / hypothesis_id / candidate_id / baseline_id'],['范围','backend / layer / category / scope'],['结果','verdict / metric / value / unit / next_test'],['溯源','evidence_refs / diagnostic / synthetic']];
  a.forEach(([label,body],i)=>{const y=228+i*127;text(s,83,y,220,52,label,32,C.green,true);text(s,337,y,1170,72,body,29,C.ink);rule(s,82,y+102,1440);});
  text(s,85,760,1430,55,'缺失证据保持 unavailable。Agent 的解释不能补成测量值。',29,C.amber,true);
}
// 10
{
  const s=slide('Profile 归因与最终性能验收','先定位开销，再关闭 profiler 做配对性能测量',
    'vLLM 官方文档提醒 profiler 会增加开销，PyTorch profiler 的 stack、memory 和 shapes 等信息不适合直接用于 benchmark。CUDA 可以按场景选择 PyTorch profiler 或 Nsight Systems。Ascend 文档区分 Ascend PyTorch Profiler 的算子观测与 MS Service Profiler 的框架函数观测。PD 场景还需要按实例采集。最终性能应在关闭 profiler 后按固定负载、环境与缓存起点做多次配对测量，保留原始日志。具体参数必须依据安装版本，本次未进行后端环境验证。',[sources.profiling,sources.ascend,sources.benchmark]);
  statement(s,83,225,670,'归因运行','CUDA：PyTorch / Nsight Systems\nAscend：PyTorch / MS Service Profiler\n输出函数、算子与通信开销证据。');
  statement(s,839,225,670,'验收运行','关闭 profiler，固定版本与负载。\n比较吞吐、p95 延迟、资源与质量。\n显式记录冷缓存或热缓存起点。');
  rule(s,83,536,1428);
  text(s,83,591,1410,100,'同一 scope 下保留 profile 与 benchmark 两类证据，\n禁止把带采样开销的 profile 数值直接发布为性能收益。',32,C.green,true);
  text(s,83,749,1410,55,'当前实现只校验反馈记录与门禁。真实探针和设备适配后续接入。',26,C.muted);
}
// 11
{
  const s=slide('分层知识库与知识晋升','结论带 scope，事实保存在证据库',
    '知识库保存结论与适用条件，证据库保存原始事实，两者通过 evidence_id 关联。system 命名空间存质量门禁、后端边界与恢复契约。组件空间覆盖 semantic-router、router、harness、agent-cache 和两条后端；后端内再使用共享 engine taxonomy。检索优先匹配模型、引擎、硬件、负载及 revision。提出、验证、否定和过期是独立状态。人工建议先进入 proposed；知识晋升要求可核查证据，不应将本轮结果倒灌为本轮先验。');
  const root=box(s,83,222,1434,85,'system：跨层契约、质量门禁、恢复规则','',C.lime,30);
  const comps=box(s,83,354,1434,130,'组件知识','semantic-router / router / harness / agent-cache\nvllm / vllm-ascend，以及 backend 内共享 taxonomy',C.white,29);connect(s,root,comps,'bottom','top');
  text(s,86,541,700,104,'scope：模型、引擎、硬件、负载\nevidence：URI / hash / revision',27,C.ink);
  text(s,855,541,660,104,'状态：proposed / verified\nrejected / deprecated',28,C.green,true);
  rule(s,83,693,1434);text(s,87,728,1420,80,'失效条件触发复验。反例与旧证据保留，已验证经验供下一轮检索。',29,C.muted);
}
// 12
{
  const s=slide('验收、激活与稳定版本','accepted、active、last_good 分别表达不同事实',
    'accepted 只表示独立裁决通过，active 表示正在服务的版本，last_good 表示观察通过的稳定版本。批准晋升时只切 active。观察失败恢复旧 last_good，恢复失败进入 NEEDS_RECOVERY 并停止新实验。后端证据必须覆盖声明的全部发布目标。当前演示的 CUDA 数字不能证明 Ascend 可用。CPU demo 可以模拟版本指针迁移，但不会部署模型服务。本页的数值与版本是演示，不能用作发布证据。');
  const a=box(s,85,248,440,145,'⑦ accepted','独立裁决通过\n当前服务版本尚未改变',C.soft,32);
  const b=box(s,585,248,440,145,'⑧ active','显式批准后激活\n保留旧 last_good',C.white,32);
  const c=box(s,1085,248,430,145,'⑨ last_good','观察通过后更新\n失败恢复稳定版本',C.lime,32);connect(s,a,b);connect(s,b,c);
  text(s,88,470,1410,93,'发布范围必须匹配证据范围。\nCUDA 通过不能自动解除 Ascend 兼容门禁。',33,C.ink,true);
  text(s,88,625,1410,110,'恢复结果未知时先对账。恢复失败进入 NEEDS_RECOVERY。\n真实部署与回滚 adapter 尚未接入。',29,C.muted);
}
// 13: A screenshot is documentary evidence of the actual local UI, not an invented UI.
let dashboardScreenshot;
try {
  await fs.access(dashboardPath);
  const { chromium }=requirePackage('playwright');
  const executable=arg('browser',process.env.PRESENTATION_BROWSER);
  const browser=await chromium.launch({headless:true,...(executable?{executablePath:executable}:{channel:'msedge'})});
  const page=await browser.newPage({viewport:{width:1440,height:980},deviceScaleFactor:1});
  await page.goto(pathToFileURL(dashboardPath).href,{waitUntil:'load'});
  await page.screenshot({path:path.join(buildDir,'dashboard.png'),fullPage:false});
  await browser.close();dashboardScreenshot=await fs.readFile(path.join(buildDir,'dashboard.png'));
} catch(error) { console.warn('Dashboard screenshot unavailable:',error.message); }
{
  const s=slide('Dashboard 的观测与人工干预','本地 mock UI。演示操作不会启动硬件实验或修改线上服务',
    'Dashboard 展示运行轮次、步骤、版本指针、候选变更、知识证据、稠密反馈、后端层次和人工干预。截图来自本地 HTML，其所有指标均为演示。R-024 当前停在⑦评估，active 与 last_good 初始同为 v0.8.2，候选 RC-024 尚未 accepted。真实部署适配不在本轮范围。可操作暂停恢复、重跑、追加假设、排除候选以及有门禁约束的晋升/回滚演示。后端真实实验结果必须独立接入。Python demo Controller 支持 next_round；HTML mock 的下一轮说明不代表创建了真实任务。');
  if(dashboardScreenshot)s.images.add({blob:dashboardScreenshot,contentType:'image/png',alt:'RSI 本地演示 Dashboard 截图，全部指标为演示',fit:'contain',position:{left:67,top:202,width:1080,height:624}});
  else{box(s,75,235,1000,450,'本地 Dashboard','运行总览 / 演化流程 / 候选与效果\n分层知识库 / 人工干预\n所有数据为演示，真实服务未连接',C.white,35);}
  text(s,1200,236,326,75,'R-024 / RC-024',31,C.green,true);
  text(s,1200,332,326,129,'⑦ 评估中\nactive v0.8.2\nlast_good v0.8.2',27,C.ink);
  text(s,1200,495,326,156,'门禁未通过\n不能批准晋升\n人工输入保留审计',27,C.ink);
  text(s,1200,697,326,77,'数字与操作均为演示',25,C.amber,true);
}
// 14
{
  const s=slide('CPU 可运行的初始 CLI','命令在仓库环境执行。输出是离线评估与持久化演示状态',
    '这些命令是本轮实现的稳定入口。taxonomy 输出后端分类；evaluate 消费 examples/rsi/feedback.json 中的显式记录并写报告；demo 在 .rsi-demo 初始化/运行离线示例；status 查看持久化状态；serve 在本机 loopback 的8877端口提供 Dashboard。本页没有提供 vllm serve 或 profiler 启动命令，因为真实后端与适配器尚未验证。默认运行前按仓库说明安装项目依赖。');
  const cmds=[['查看分类','python -m agentinfer.rsi taxonomy --backend cuda'],['评估反馈','python -m agentinfer.rsi evaluate examples/rsi/feedback.json --output report.json'],['初始化演示','python -m agentinfer.rsi demo --run-dir .rsi-demo'],['读取状态','python -m agentinfer.rsi status --run-dir .rsi-demo'],['打开运行台','python -m agentinfer.rsi serve --run-dir .rsi-demo --port 8877']];
  cmds.forEach(([label,cmd],i)=>{const y=211+i*114;text(s,82,y,220,55,label,28,C.green,true);text(s,302,y,1215,75,cmd,i===1?23:25,C.ink);if(i<4)rule(s,82,y+95,1435);});
}
// 15
{
  const s=slide('本轮代码的实现边界','offline scaffold 提供可执行契约，真实系统接入保留明确边界',
    '初始实现由 Python 控制状态、持久化 demo、taxonomy、显式 FeedbackRecords 评估、scoped knowledge 和 loopback Dashboard 组成。它是能够在 CPU 环境检查合同的脚手架。本次不会运行真实 ZCode、AgentBench、GPU/NPU profiler 或模型部署。未来 integrations 应调用已有 runtime/benchmark，不应复制 AgentBench 运行器、proxy 和指标计算。两条后端是否支持模型与插件仍须独立环境验证。');
  statement(s,84,224,680,'本轮已提供','Python CLI 与持久化演示 Controller\n显式反馈评估与分层知识记录\n本地 Dashboard 和文档资产');
  statement(s,855,224,660,'后续接入','真实 ZCode / openJiuwen runtime\nAgentBench 与 profiler adapter\n设备租约、部署、观察与恢复');
  rule(s,84,552,1434);
  text(s,86,600,1430,96,'Controller 持有状态机。Harness 与 integrations 报告事实。\nAgentBench 保留既有 runtime、replay 与评测能力。',31,C.green,true);
  text(s,86,761,1420,43,'CUDA 和 vLLM-Ascend 均未进行环境验证，性能收益尚无真实测量。',26,C.amber,true);
}
// 16
{
  const s=slide('接入路线与验收条件','每次只增加一个可验证边界',
    '后续建议按风险和依赖推进。先接一个真实 ZCode session，验证流式、工具调用、超时和取消；再让 AgentBench 驱动固定 workload，区分 replay 性能与真实任务正确性；然后接局部探针，验证反馈记录能够回溯原始产物；最后实现设备隔离和受控发布恢复。Ascend 是独立后端验证矩阵，不能复制 CUDA 通过结论。这里是后续路线，不构成本次实现声明。',[sources.zcodeCli,sources.replay,sources.benchmark,sources.ascend]);
  const phases=[['1','Agent 接入','真实 session、tool call、取消与恢复'],['2','实验接入','固定负载，replay 与真实任务分别验收'],['3','局部探针','参考 / 差分 / 干预与 profile 可回溯'],['4','受控发布','后端矩阵、健康观察、失败对账与恢复']];
  phases.forEach(([n,t,b],i)=>{const y=219+i*140;text(s,84,y,85,66,n,53,C.green,true);text(s,205,y,315,59,t,33,C.ink,true);text(s,552,y,954,83,b,29,C.muted);if(i<3)rule(s,84,y+113,1430);});
}

const architectureMermaid=`flowchart TB
  app[应用与 Agents] --> harness[Harness：角色、DAG、会话与工具]
  harness --> infer[AgentInfer：语义路由、请求路由、准入与缓存观测]
  infer --> cuda[vLLM / CUDA]
  infer --> ascend[vLLM-Ascend]
  control[RSI Controller：唯一状态机、预算与恢复] -. 派发 .-> harness
  control --> lab[隔离实验与独立裁决]
  lab --> evidence[原始证据库]
  evidence --> kb[分层知识库]
  infer -. 运行观测 .-> evidence
  kb -. 下一轮适用经验 .-> harness
  lab -. 经门禁受控晋升 .-> infer
  note[固定模型权重；两后端均未环境验证；真实适配后续接入]
`;
const feedbackMermaid=`flowchart LR
  H[③可证伪假设] --> P[④⑤参考/差分/受控干预/消融+profile]
  P --> F[⑥结构化 FeedbackRecord]
  F --> L[④局部修正]
  L --> P
  L --> E[⑤集成与真实任务E2E]
  E --> G{⑦独立裁决}
  G -->|PASS| A[⑧激活恢复]
  A --> O[⑨观察]
  O --> K[⑩知识归档与晋升]
  G -->|FAIL| H
  G -->|INCONCLUSIVE：目标增量重测| P
  G -. 当前脚手架重试均返回③ .-> H
  O -->|异常| A
  K --> N[①下一轮基线 / ②诊断]
  N --> H
  K -. scope匹配知识 .-> H
  C[三类反馈：correctness / system_behavior / performance]
  C -. 顺序由假设决定 .-> P
  R[局部、便宜、及时、客观可验证；profile结果不作最终perf验收]
`;
await fs.writeFile(path.join(outDir,'rsi-layered-architecture.mmd'),architectureMermaid,'utf8');
await fs.writeFile(path.join(outDir,'rsi-dense-feedback.mmd'),feedbackMermaid,'utf8');
await fs.writeFile(path.join(outDir,'rsi-layered-architecture.svg'),svgDiagram(architecture,'整体分层架构'),'utf8');
await fs.writeFile(path.join(outDir,'rsi-dense-feedback.svg'),svgDiagram(feedbackDiagram,'稠密反馈与知识回路'),'utf8');

const candidatePath=path.join(buildDir,'candidate.pptx');
await (await PresentationFile.exportPptx(presentation)).save(candidatePath);
const finalPath=path.join(buildDir,`final-${Date.now()}`,'agentinfer-rsi-guide-zh.pptx');
await fs.mkdir(path.dirname(finalPath),{recursive:true});
const receiptPath=path.join(buildDir,`validation-${path.basename(path.dirname(finalPath))}.json`);
const result=await finalizePresentation({workspaceDir:buildDir,candidatePath,finalPath,pythonExecutable:python,
  integrityValidatorPath:path.join(skillDir,'container_tools/inspect_presentation_package_integrity.py'),
  layoutValidatorPath:path.join(skillDir,'container_tools/inspect_presentation_layout_geometry.py'),
  layoutArgs:['--expected-slide-size-emu','15240000,8572500','--validate-bullet-geometry','--validate-heading-fit'],
  explicitTotalSlideCount:16,requiredNativeTableOwnerSlides:[],requiredNativeChartOwnerSlides:[],
  fontPolicy:{basis:'design',families:[family]},verifyArtifactToolImport:true,receiptPath});
console.log(JSON.stringify({finalPath,receiptPath,finalized:!!result}));
const checked=await PresentationFile.importPptx(await FileBlob.load(finalPath));
const renderDir=path.join(buildDir,'renders');await fs.mkdir(renderDir,{recursive:true});
for(let i=0;i<16;i++){
  const s=checked.slides.getItem(i);
  const preview=await checked.export({slide:s,format:'png',scale:1});
  const bytes=new Uint8Array(await preview.arrayBuffer());
  await fs.writeFile(path.join(renderDir,`slide-${String(i+1).padStart(2,'0')}.png`),bytes);
  if(i===2)await fs.writeFile(path.join(outDir,'rsi-layered-architecture.png'),bytes);
  if(i===3)await fs.writeFile(path.join(outDir,'rsi-backend-layers.png'),bytes);
  if(i===5)await fs.writeFile(path.join(outDir,'rsi-numbered-loop.png'),bytes);
  if(i===6)await fs.writeFile(path.join(outDir,'rsi-dense-feedback.png'),bytes);
  const layout=await s.export({format:'layout'});await fs.writeFile(path.join(renderDir,`slide-${i+1}.layout.json`),await layout.text());
}
const montage=await checked.export({format:'png',montage:{columns:4,slideWidth:400,gap:12,padding:16,background:C.bg}});
await fs.writeFile(path.join(buildDir,'montage.png'),new Uint8Array(await montage.arrayBuffer()));
await fs.copyFile(finalPath,path.join(outDir,'agentinfer-rsi-guide-zh.pptx'));
console.log(JSON.stringify({slides:16,output:path.join(outDir,'agentinfer-rsi-guide-zh.pptx'),renderDir}));
