import { readFile } from 'node:fs/promises';
import vm from 'node:vm';

const html = await readFile(new URL('./yieldmindx-five-section-workspace-chart-recovery.html', import.meta.url), 'utf8');
const asset = await readFile(new URL('./yieldmindx-commonality-parameters.js', import.meta.url), 'utf8');
const gvbText = await readFile('C:\\personal\\קורס AI\\YieldMindX\\Data\\GVB.csv', 'utf8');

const inlineScripts = [...html.matchAll(/<script(?:\s[^>]*)?>([\s\S]*?)<\/script>/gi)].map((match) => match[1]).filter((script) => script.trim());
for (const script of inlineScripts) new Function(script);
new Function(asset);

const script = inlineScripts.at(-1);
const coreStart = script.indexOf('const trim=');
const coreEnd = script.indexOf("const stages=");
if (coreStart < 0 || coreEnd < 0) throw new Error('Analysis core was not found.');
const core = script.slice(coreStart, coreEnd);
const sandbox = { window: {}, gvbText };
sandbox.window = sandbox;
vm.createContext(sandbox);
vm.runInContext(asset, sandbox);
vm.runInContext(`${core};globalThis.answer=analyse(csv(gvbText),csv(window.YMX_PARAMETER_DATA.csv));`, sandbox, { timeout: 120000 });
const result = sandbox.answer;
const summary = {
  matched: result.matched,
  good: result.good,
  bad: result.bad,
  tested: result.out.length,
  qBelow05: result.out.filter((row) => row.q < .05).length,
  minP: Math.min(...result.out.map((row) => row.p)),
  minQ: Math.min(...result.out.map((row) => row.q)),
  gvbOnly: result.gvbOnly,
  parameterOnly: result.parameterOnly,
  oneFileInput: (html.match(/<input\b[^>]*type="file"/gi) || []).length,
  oldLabelsAbsent: !/GVB outcome CSV|Parameter measurement CSV|Select both CSV files/i.test(html),
  evidenceFieldsPresent: ['Good sample SD','Bad sample SD','95% CI','Welch t','Welch df','good_sd','bad_sd','welch_t','welch_df'].every((label) => html.includes(label)),
  resultFieldsPresent: result.out.every((row) => ['sg','sb','lo','hi','t','df'].every((field) => Number.isFinite(row[field])))
};
const expected = { matched: 1305, good: 1014, bad: 291, tested: 1404, qBelow05: 0, gvbOnly: 0, parameterOnly: 195, oneFileInput: 1, oldLabelsAbsent: true, evidenceFieldsPresent: true, resultFieldsPresent: true };
for (const [key, value] of Object.entries(expected)) if (summary[key] !== value) throw new Error(`${key}: expected ${value}, got ${summary[key]}`);
console.log(JSON.stringify(summary, null, 2));
