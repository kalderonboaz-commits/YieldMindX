import { createHash } from 'node:crypto';
import { readFile, writeFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

const here = path.dirname(fileURLToPath(import.meta.url));
const sourcePath = 'C:\\personal\\קורס AI\\YieldMindX\\Data\\Commnality_paramters.csv';
const dashboardPath = path.join(here, 'yieldmindx-five-section-workspace-chart-recovery.html');
const tailPath = path.join(here, 'gvb-section-tail.html');
const assetPath = path.join(here, 'yieldmindx-commonality-parameters.js');

function parseCsv(text) {
  const rows = [];
  let row = [], value = '', quoted = false;
  for (let i = 0; i < text.length; i += 1) {
    const char = text[i];
    if (quoted) {
      if (char === '"' && text[i + 1] === '"') { value += '"'; i += 1; }
      else if (char === '"') quoted = false;
      else value += char;
    } else if (char === '"') quoted = true;
    else if (char === ',') { row.push(value); value = ''; }
    else if (char === '\n') { row.push(value.replace(/\r$/, '')); rows.push(row); row = []; value = ''; }
    else value += char;
  }
  row.push(value.replace(/\r$/, ''));
  if (row.some((cell) => cell !== '')) rows.push(row);
  if (quoted || rows.length < 2) throw new Error('Parameter CSV is malformed.');
  return rows;
}

const csv = await readFile(sourcePath, 'utf8');
const rows = parseCsv(csv);
const headers = rows[0].map((value, index) => value.replace(/^\uFEFF/, '').trim() || `Column ${index + 1}`);
const lotIndex = headers.indexOf('LotName');
const waferIndex = headers.indexOf('WaferNum');
if (lotIndex < 0 || waferIndex < 0) throw new Error('Fixed parameter data must contain LotName and WaferNum.');
if (new Set(headers).size !== headers.length) throw new Error('Fixed parameter data contains duplicate headers.');

let missingKeys = 0, duplicateKeys = 0;
const keys = new Set();
for (let index = 1; index < rows.length; index += 1) {
  const lot = (rows[index][lotIndex] ?? '').trim();
  const wafer = (rows[index][waferIndex] ?? '').trim();
  if (!lot || !wafer) { missingKeys += 1; continue; }
  const key = `${lot}\u0000${wafer}`;
  if (keys.has(key)) duplicateKeys += 1;
  keys.add(key);
}
if (missingKeys || duplicateKeys) throw new Error(`KING DAVID validation failed: ${missingKeys} missing keys; ${duplicateKeys} duplicate keys.`);

const meta = {
  name: 'Commnality_paramters.csv',
  rows: rows.length - 1,
  columns: headers.length,
  missingKeys,
  duplicateKeys,
  joinKey: ['LotName', 'WaferNum'],
  sha256: createHash('sha256').update(csv).digest('hex'),
  validated: true
};
const asset = `window.YMX_PARAMETER_DATA=Object.freeze({csv:${JSON.stringify(csv)},meta:Object.freeze(${JSON.stringify(meta)})});\n`;
await writeFile(assetPath, asset, 'utf8');

const dashboard = await readFile(dashboardPath, 'utf8');
const tail = await readFile(tailPath, 'utf8');
const marker = '<section class="section" id="gvb">';
const start = dashboard.indexOf(marker);
if (start < 0) throw new Error('GVB section marker was not found in the dashboard.');
await writeFile(dashboardPath, dashboard.slice(0, start) + tail, 'utf8');

console.log(JSON.stringify({ assetPath, dashboardPath, meta }, null, 2));
