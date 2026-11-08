const { makeBrowser, until, PAYLOAD, assert } = require("./harness.js");
const results = [];
const test = async (name, fn) => { try { await fn(); results.push("PASS  " + name); } catch (e) { results.push("FAIL  " + name + "\n      " + (e.stack || e).split("\n").slice(0, 4).join("\n      ")); } };
const J = (status, body, extra = {}) => ({ status, body, ...extra });
const health = { status: "ok", ollama: { reachable: true, model: "qwen2.5:7b", model_available: true }, documents: 1, chunks: 12, jobs_pending: 0, index_error: null, max_batch_fields: 5 };
const FIELDS = ["loan_tenure", "loan_amount", "applicant_age"];
const ROUTES = ["verified", "error", "not_found"];
const mkPayload = (name, route) => ({ ...PAYLOAD, field: name, batch_field: name, route, answer: "answer for " + name, session_id: undefined, turn: undefined });
const describeBatch = (state) => {
  const done = state.done, status = state.status;
  return { batch_id: "b".repeat(32), status, total: 3, done, current: status === "running" ? FIELDS[done] : null, fields: FIELDS,
    items: FIELDS.slice(0, done).map((f, i) => ({ field: f, route: ROUTES[i], headline: i === 0 ? "maximum 30 years" : "", seconds: 100 + i })),
    created_at: "2026-03-04T09:00:00+00:00", finished_at: status === "running" ? null : "x", elapsed_seconds: 300, options: {} };
};
function fakeServer(initial = { status: "running", done: 1 }) {
  const state = { ...initial, calls: [], hold: false, advance: 0, exportResponse: null };
  const handler = async (path, o = {}) => {
    const m = o.method || "GET"; state.calls.push(m + " " + path);
    if (path === "/health") return J(200, health);
    if (path === "/documents") return J(200, { documents: [], total_chunks: 0 });
    if (path === "/batches" && m === "POST") { state.posted = JSON.parse(o.body); return J(202, { batch_id: "b".repeat(32), job_id: "j1", count: state.posted.fields.length }); }
    if (path === "/batches/" + "b".repeat(32)) {
      if (state.gone) return J(404, { detail: "Unknown batch." });
      if (state.status === "running" && state.done >= 3) state.status = "done";
      const body = describeBatch(state);
      const canMove = !state.hold || state.advance > 0;
      if (canMove && state.status === "running") { state.done = Math.min(3, state.done + 1); if (state.hold) state.advance--; }   // one field per poll
      return J(200, body);
    }
    if (path === "/batches/" + "b".repeat(32) + "/results") return J(200, { results: FIELDS.map((f, i) => mkPayload(f, ROUTES[i])) });
    if (path === "/batches/" + "b".repeat(32) + "/cancel") { state.status = "cancelled"; return J(200, describeBatch(state)); }
    if (path.startsWith("/batches/" + "b".repeat(32) + "/export")) return state.exportResponse || J(200, null, { blob: "XLSXBYTES", headers: { "Content-Disposition": 'attachment; filename="lending-policy-test-values_2026-03-04_0930.xlsx"' } });
    if (path.startsWith("/sessions/")) return J(404, { detail: "nope" });
    return J(404, { detail: "nope " + path });
  };
  return { handler, state };
}
const text = (b) => b.ids["batch-area"].textContent;
const allEls = (b) => b.ids["batch-area"].all();
const button = (b, label) => allEls(b).find((e) => e.tag === "button" && e.textContent.startsWith(label));

module.exports = async () => {
  await test("tabs switch between Ask and Batch and keep aria state", async () => {
    const b = makeBrowser({ server: fakeServer().handler }); await until(() => b.ids.status.textContent.includes("ready"), "load");
    b.ids["tab-batch"].onclick();
    assert.strictEqual(b.ids["batch-panel"].hidden, false); assert.strictEqual(b.ids["ask-panel"].hidden, true);
    assert.strictEqual(b.ids["tab-batch"].getAttribute("aria-selected"), "true"); assert.strictEqual(b.ids["tab-ask"].getAttribute("aria-selected"), "false");
    b.ids["tab-ask"].onclick(); assert.strictEqual(b.ids["ask-panel"].hidden, false);
  });

  await test("live count: bullets, repeats, the server limit, and the Run button state", async () => {
    const b = makeBrowser({ server: fakeServer().handler }); await until(() => b.ids.status.textContent.includes("ready"), "load");
    const type = (t) => { b.ids["batch-fields"].value = t; b.ids["batch-fields"].oninput(); };
    assert.ok(b.ids["batch-count"].textContent.startsWith("0 fields") && b.ids["batch-run"].disabled);
    type("- loan_tenure\n2. loan_amount, Loan_Tenure; applicant_age");
    assert.ok(b.ids["batch-count"].textContent.startsWith("3 fields") && b.ids["batch-count"].textContent.includes("1 repeated removed"));
    assert.strictEqual(b.ids["batch-run"].disabled, false);
    type("a\nb\nc\nd\ne\nf\ng");
    assert.ok(b.ids["batch-count"].textContent.includes("7 fields") && b.ids["batch-count"].textContent.includes("over the limit of 5: remove 2"));
    assert.strictEqual(b.ids["batch-run"].disabled, true);
    type("x".repeat(501)); assert.ok(b.ids["batch-count"].textContent.includes("longer than 500")); assert.strictEqual(b.ids["batch-run"].disabled, true);
    type("one"); assert.ok(b.ids["batch-count"].textContent.startsWith("1 field ")  || b.ids["batch-count"].textContent === "1 field");
  });

  await test("loading a .txt file appends its fields, read in the browser", async () => {
    const b = makeBrowser({ server: fakeServer().handler }); await until(() => b.ids.status.textContent.includes("ready"), "load");
    b.ids["batch-fields"].value = "loan_tenure";
    b.ids["batch-file"].files = [{ name: "fields.txt", text: async () => "loan_amount\napplicant_age\n" }];
    await b.ids["batch-file"].onchange();
    assert.strictEqual(b.ids["batch-fields"].value, "loan_tenure\nloan_amount\napplicant_age\n");
    assert.ok(b.ids["batch-count"].textContent.startsWith("3 fields"));
    assert.ok(!b.calls.some((c) => c.method === "POST"), "nothing was uploaded");
  });

  await test("run a batch: progress bar, current field, estimate, per-field list, then finished with downloads and expandable fields", async () => {
    const s = fakeServer({ status: "running", done: 0 }); s.state.hold = true; s.state.posted = null;
    const b = makeBrowser({ server: s.handler }); await until(() => b.ids.status.textContent.includes("ready"), "load");
    b.ids["tab-batch"].onclick(); b.ids["batch-fields"].value = "loan_tenure\nloan_amount\napplicant_age"; b.ids["batch-fields"].oninput();
    b.ids.kinds.value = "date"; b.ids.fast.checked = true; b.ids.source.value = "";
    const running = b.ids["batch-run"].onclick();
    await until(() => text(b).includes("now: loan_tenure"), "progress");
    assert.deepStrictEqual(s.state.posted, { fields: FIELDS, source_file: null, kinds: "date", fast: true });
    assert.strictEqual(b.store.get("lpa.batch"), "b".repeat(32), "batch id remembered for a reload");
    assert.strictEqual(b.ids.send.disabled, true, "asking is blocked while the batch runs");
    const bar = allEls(b).find((e) => e.attrs.role === "progressbar"); assert.strictEqual(bar.attrs["aria-valuemax"], "3"); assert.strictEqual(bar.attrs["aria-valuenow"], "0");
    assert.ok(button(b, "Stop after current field"));
    s.state.advance = 1;                                                              // exactly one field finishes
    await until(() => text(b).includes("1 of 3 done"), "after one field");
    assert.ok(/about .* left/.test(text(b)), "remaining-time estimate appears once a field has finished");
    assert.ok(text(b).includes("maximum 30 years") && text(b).includes("Verified"));
    assert.strictEqual(allEls(b).find((e) => e.attrs.role === "progressbar").attrs["aria-valuenow"], "1");
    s.state.hold = false;
    await running;
    assert.ok(text(b).includes("Finished: 3 of 3 fields"));
    for (const label of ["Verified: 1", "Error: 1", "Not found: 1", "Needs review: 0"]) assert.ok(text(b).includes(label), label);
    assert.ok(button(b, "Download Excel (all fields)") && button(b, "Download CSV") && button(b, "Close") && button(b, "Delete batch"));
    assert.ok(!button(b, "Stop after current field"));
    assert.strictEqual(b.ids.send.disabled, false);
    const details = allEls(b).filter((e) => e.tag === "details" && e.children[0].textContent.includes("  ·  "));   // one per field (each answer card has its own inner details)
    assert.strictEqual(details.length, 3); assert.ok(details[0].textContent.includes("loan_tenure") && details[0].textContent.includes("answer for loan_tenure"));
  });

  await test("Download Excel fetches the export and saves it under the server's file name", async () => {
    const s = fakeServer({ status: "done", done: 3 }); const b = makeBrowser({ server: s.handler, storage: { "lpa.batch": "b".repeat(32) } });
    await until(() => button(b, "Download Excel"), "finished card");
    await button(b, "Download Excel").onclick();
    assert.ok(s.state.calls.includes("GET /batches/" + "b".repeat(32) + "/export?format=xlsx"));
    assert.strictEqual(b.downloads.length, 1); assert.strictEqual(b.downloads[0].download, "lending-policy-test-values_2026-03-04_0930.xlsx");
    await button(b, "Download CSV").onclick(); assert.ok(s.state.calls.some((c) => c.endsWith("export?format=csv")));
  });

  await test("a refused download shows the server's reason instead of an error page", async () => {
    const s = fakeServer({ status: "done", done: 3 }); s.state.exportResponse = J(404, { detail: "This batch has no finished fields yet, so there is nothing to export." });
    const b = makeBrowser({ server: s.handler, storage: { "lpa.batch": "b".repeat(32) } });
    await until(() => button(b, "Download Excel"), "card"); await button(b, "Download Excel").onclick();
    assert.ok(b.ids.notice.textContent.includes("nothing to export")); assert.strictEqual(b.downloads.length, 0);
  });

  await test("RELOAD during a batch: the Batch tab opens and the batch is followed to the end", async () => {
    const s = fakeServer({ status: "running", done: 1 });
    const b = makeBrowser({ server: s.handler, storage: { "lpa.batch": "b".repeat(32) } });
    await until(() => text(b).includes("Finished"), "resumed batch finished");
    assert.strictEqual(b.ids["batch-panel"].hidden, false); assert.strictEqual(b.ids.send.disabled, false);
  });

  await test("a batch the server forgot: clear message and the saved id is dropped", async () => {
    const s = fakeServer(); s.state.gone = true; const b = makeBrowser({ server: s.handler, storage: { "lpa.batch": "b".repeat(32) } });
    await until(() => text(b).includes("no longer has this batch"), "message"); assert.strictEqual(b.store.get("lpa.batch"), undefined);
    assert.strictEqual(b.ids.send.disabled, false);
  });

  await test("Stop after current field sends the cancel request and the batch shows as stopped", async () => {
    const s = fakeServer({ status: "running", done: 1 }); s.state.hold = true;
    const b = makeBrowser({ server: s.handler, storage: { "lpa.batch": "b".repeat(32) } });
    await until(() => button(b, "Stop after current field"), "stop button");
    const stop = button(b, "Stop after current field"); await stop.onclick();
    assert.ok(s.state.calls.includes("POST /batches/" + "b".repeat(32) + "/cancel")); assert.ok(stop.disabled);
    await until(() => text(b).includes("Stopped: 1 of 3") || text(b).includes("Stopped:"), "stopped");
    assert.ok(text(b).includes("the rest were skipped"));
  });

  await test("Close forgets the batch; Delete asks first and calls the server", async () => {
    const s = fakeServer({ status: "done", done: 3 }); s.handler0 = s.handler;
    const server = async (p, o = {}) => ((o.method === "DELETE" && p.startsWith("/batches/")) ? (s.state.calls.push("DELETE " + p), J(200, { deleted: "x" })) : s.handler(p, o));
    const b = makeBrowser({ server, storage: { "lpa.batch": "b".repeat(32) }, confirmAnswer: true });
    await until(() => button(b, "Close"), "card"); button(b, "Delete batch").onclick();
    await until(() => s.state.calls.includes("DELETE /batches/" + "b".repeat(32)), "delete call"); await until(() => text(b) === "", "area cleared");
    assert.strictEqual(b.store.get("lpa.batch"), undefined);
    const b2 = makeBrowser({ server: s.handler, storage: { "lpa.batch": "b".repeat(32) }, confirmAnswer: false });
    await until(() => button(b2, "Delete batch"), "card2"); button(b2, "Delete batch").onclick(); await new Promise((r) => setTimeout(r, 20));
    assert.ok(text(b2).includes("Finished")); assert.strictEqual(b2.store.get("lpa.batch"), "b".repeat(32), "declined: nothing deleted");
  });

  await test("hostile field names and headlines in a batch stay text", async () => {
    const s = fakeServer({ status: "done", done: 3 }); const evil = "<img src=x onerror=alert(1)>";
    const server = async (p, o) => { const r = await s.handler(p, o); if (p === "/batches/" + "b".repeat(32) && r.body) { r.body.fields = [evil, "b", "c"]; r.body.items[0].headline = "<script>x</script>"; } return r; };
    const b = makeBrowser({ server, storage: { "lpa.batch": "b".repeat(32) } });
    await until(() => text(b).includes("Finished"), "card"); assert.ok(text(b).includes(evil));
    assert.ok(!allEls(b).some((e) => ["img", "script"].includes(e.tag)));
  });

  await test("single answers get Test values (.xlsx); the conversation gets Export all answers", async () => {
    const answer = { ...PAYLOAD };                                                    // has session_id, turn 1 and findings
    const state = { calls: [] };
    const server = async (path, o = {}) => {
      state.calls.push((o.method || "GET") + " " + path);
      if (path === "/health") return J(200, health); if (path === "/documents") return J(200, { documents: [], total_chunks: 0 });
      if (path === "/chat?wait=false") return J(202, { job_id: "chat1", session_id: "sess12345678" });
      if (path === "/jobs/chat1") return J(200, { job_id: "chat1", status: "done", result: answer });
      if (path.includes("/export")) return J(200, null, { blob: "X", headers: { "Content-Disposition": 'attachment; filename="f.xlsx"' } });
      return J(404, { detail: "x" });
    };
    const b = makeBrowser({ server }); await until(() => b.ids.log.textContent.includes("Ask for a field"), "load");
    assert.strictEqual(b.ids["export-conv"].hidden, true);
    b.ids.message.value = "loan_tenure"; await b.ids.form.onsubmit({ preventDefault() {} });
    const buttons = b.ids.log.all().filter((e) => e.tag === "button").map((e) => e.textContent);
    assert.deepStrictEqual(buttons, ["Report (.md)", "Test values (.xlsx)"]);
    await b.ids.log.all().find((e) => e.textContent === "Test values (.xlsx)").onclick();
    assert.ok(state.calls.includes("GET /sessions/sess12345678/turns/1/export?format=xlsx"));
    assert.strictEqual(b.ids["export-conv"].hidden, false);
    await b.ids["export-conv"].onclick(); assert.ok(state.calls.includes("GET /sessions/sess12345678/export?format=xlsx"));
    assert.strictEqual(b.downloads.length, 2);
  });

  return results;
};
