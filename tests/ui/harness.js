// Test harness for the web page: a very small simulated browser (fake DOM, localStorage, fetch). No packages needed.
// A very small simulated browser, just enough for the page: fake DOM, localStorage, fetch.
const fs = require("fs"), vm = require("vm"), assert = require("assert");
const path = require("path");
const html = fs.readFileSync(path.join(__dirname, "..", "..", "src", "lending_agent", "api", "static", "index.html"), "utf8");
const script = html.match(/<script>([\s\S]*)<\/script>/)[1];
const realSetTimeout = setTimeout;
class Text { constructor(t) { this.text = String(t); this.parent = null; } get textContent() { return this.text; } }
class El {
  constructor(tag) { this.tag = tag; this.children = []; this.parent = null; this.props = {}; this._text = ""; this.className = ""; this.hidden = false; this.disabled = false; this.value = ""; this.files = []; this.style = {}; this.attrs = {}; }
  get textContent() { return this._text + this.children.map((c) => c.textContent).join(""); }
  set textContent(v) { this._text = String(v); this.children = []; }
  append(...nodes) { for (const n of nodes) { const node = typeof n === "string" ? new Text(n) : n; node.parent = this; this.children.push(node); } }
  replaceChildren(...nodes) { this.children = []; this._text = ""; this.append(...nodes); }
  replaceWith(node) { const i = this.parent.children.indexOf(this); node.parent = this.parent; this.parent.children[i] = node; }
  remove() { if (this.parent) this.parent.children = this.parent.children.filter((c) => c !== this); }
  get options() { return this.children; }
  all() { return this.children.flatMap((c) => (c instanceof El ? [c, ...c.all()] : [])); }
  querySelector(sel) { const cls = sel.slice(1); return this.all().find((e) => e.className.split(" ").includes(cls)) || null; }
  setAttribute(k, v) { this.attrs[k] = v; } getAttribute(k) { return this.attrs[k]; } scrollIntoView() {} focus() {} click() { if (this.tag === "a") this.onDownload(this); }
}
function makeBrowser({ server, storage = {}, confirmAnswer = true }) {
  const ids = {}; for (const m of html.matchAll(/id="([^"]+)"/g)) ids[m[1]] = new El("x");
  ids.fast.checked = false; ids.kinds.value = "both"; ids.source.value = "";
  const body = new El("body"), downloads = [], calls = [];
  const store = new Map(Object.entries(storage));
  const ctx = {
    document: { getElementById: (id) => ids[id], createElement: (t) => { const e = new El(t); if (t === "a") e.onDownload = (a) => downloads.push(a); return e; }, body },
    localStorage: { getItem: (k) => (store.has(k) ? store.get(k) : null), setItem: (k, v) => store.set(k, String(v)), removeItem: (k) => store.delete(k) },
    fetch: async (path, options = {}) => { calls.push({ path, method: options.method || "GET", body: options.body }); const r = await server(path, options); return { ok: r.status < 400, status: r.status, statusText: "x", json: async () => r.body, headers: { get: (k) => (r.headers || {})[k] || null }, blob: async () => r.blob }; },
    setTimeout: (fn) => realSetTimeout(fn, 0), setInterval: () => 0, confirm: () => confirmAnswer,
    FormData: class { constructor() { this.parts = []; } append(k, v) { this.parts.push([k, v]); } }, Blob: class { constructor(parts) { this.text = parts.join(""); } },
    URL: { createObjectURL: (b) => { ctx.lastBlob = b; return "blob:x"; }, revokeObjectURL() {} }, console, Date, Promise, JSON, Math, String, Object, Array, Error,
  };
  vm.createContext(ctx); vm.runInContext(script, ctx);
  return { ctx, ids, store, downloads, calls, run: (code) => vm.runInContext(code, ctx) };
}
async function until(cond, label) { for (let i = 0; i < 400; i++) { if (cond()) return; await new Promise((r) => realSetTimeout(r, 5)); } throw new Error("timed out waiting for: " + label); }
const PAYLOAD = { route: "verified", answer: "For loan_tenure: maximum 30 years, per policy.pdf, page 12.", field: "loan_tenure", min_value: null, max_value: 30, unit: "years",
  cutoff_date: null, date_direction: null, date_precision: null, source_quote: "The repayment period shall not extend beyond thirty (30) years.", source_file: "policy.pdf", page_number: 12,
  review_reasons: [], conflicts: [], findings: [{ kind: "numeric_boundary", bound_type: "maximum", value: 30, unit: "years", classification: "firm_rule", grounded: true, source_file: "policy.pdf", page_number: 12, quote: "x | y\nz", applies_to: null }],
  rejected: [], trace: [{ node: "extract", seconds: 1.2 }], attempts: 1, widened_search: false, elapsed_seconds: 75, query_used: "loan_tenure", rewritten: false, session_id: "sess12345678", turn: 1 };
module.exports = { makeBrowser, until, PAYLOAD, assert, El };
