// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

// Expand the prompt only inside the final Node process, avoiding OS argv limits.
const fs = require("node:fs");
const file = process.env.AGENTBENCH_ZCODE_PROMPT_FILE;
const index = process.argv.indexOf("--prompt");
if (file && index >= 0 && process.argv[index + 1] === file) {
  const prompt = fs.readFileSync(file, "utf8");
  process.argv[index + 1] = prompt;
  fs.writeFileSync(process.env.AGENTBENCH_ZCODE_PROMPT_RECEIPT, String(Buffer.byteLength(prompt, "utf8")));
  delete process.env.AGENTBENCH_ZCODE_PROMPT_FILE;
  delete process.env.AGENTBENCH_ZCODE_PROMPT_RECEIPT;
  delete process.env.NODE_OPTIONS;
}
