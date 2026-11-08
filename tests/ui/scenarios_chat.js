const { makeBrowser, until, PAYLOAD, assert } = require("./harness.js");
const results = [];
const test = async (name, fn) => { try { await fn(); results.push("PASS  " + name); } catch (e) { results.push("FAIL  " + name + "\n      " + (e.stack || e).split("\n").slice(0, 3).join("\n      ")); } };
const J = (status, body) => ({ status, body });
const docsBody = { documents: [{ name: "policy.pdf", chunks: 12 }], total_chunks: 12 };
const health = { status: "ok", ollama: { reachable: true, model: "qwen2.5:7b", model_available: true }, documents: 1, chunks: 12, jobs_pending: 0, index_error: null };

// a fake server whose chat job takes `slowPolls` polls to finish
function fakeServer({ slowPolls = 2, payload = PAYLOAD, sessions = {}, jobs = {} } = {}) {
  const state = { polls: 0, sessions, jobs, deleted: [] };
  const handler = async (path, o = {}) => {
    const m = o.method || "GET";
    if (path === "/health") return J(200, health);
    if (path === "/documents" && m === "GET") return J(200, docsBody);
    if (path === "/chat?wait=false") {
      const req = JSON.parse(o.body); state.lastChat = req;
      const sid = req.session_id || "sess12345678";
      state.jobs.chat1 = { job_id: "chat1", status: "queued", jobs_ahead: 0, running_seconds: 0, result: null, error: null };
      state.sessions[sid] = state.sessions[sid] || { session_id: sid, turns: [] };
      return J(202, { job_id: "chat1", status: "queued", session_id: sid });
    }
    if (path.startsWith("/jobs/")) {
      const job = state.jobs[path.slice(6)]; if (!job) return J(404, { detail: "Unknown or expired job." });
      if (job.job_id === "chat1" && job.status !== "done" && job.status !== "failed") {
        state.polls++;
        if (state.polls > slowPolls && !state.hold) {
          job.status = "done"; job.result = { ...payload, session_id: state.lastChat ? (state.lastChat.session_id || "sess12345678") : "sess12345678" };
          state.sessions[job.result.session_id].turns.push({ turn: 1, message: state.lastChat ? state.lastChat.message : "loan_tenure", payload: job.result });
        } else { job.status = "running"; job.running_seconds = state.polls * 2; }
      }
      return J(200, job);
    }
    if (path.startsWith("/sessions/")) {
      const sid = decodeURIComponent(path.slice(10));
      if (m === "DELETE") { state.deleted.push(sid); delete state.sessions[sid]; return J(200, { deleted: true }); }
      return state.sessions[sid] ? J(200, state.sessions[sid]) : J(404, { detail: "Unknown or expired session." });
    }
    return J(404, { detail: "nope" });
  };
  return { handler, state };
}
const text = (b) => b.ids.log.textContent;
const ask = (b, message) => { b.ids.message.value = message; return b.ids.form.onsubmit({ preventDefault() {} }); };

module.exports = async () => {
  await test("fresh load shows the documents, the hint and an enabled Ask button", async () => {
    const s = fakeServer(); const b = makeBrowser({ server: s.handler });
    await until(() => text(b).includes("Ask for a field"), "hint");
    assert.ok(b.ids.docs.textContent.includes("policy.pdf") && b.ids.docs.textContent.includes("12 chunks"));
    assert.strictEqual(b.ids.send.disabled, false);
    assert.ok(b.ids.status.textContent.includes("ready (qwen2.5:7b)"));
  });

  await test("asking shows the pending state, then a Verified answer, and records the conversation", async () => {
    const s = fakeServer({ slowPolls: 0 }); s.state.hold = true; const b = makeBrowser({ server: s.handler });
    await until(() => text(b).includes("Ask for a field"), "load");
    const done = ask(b, "loan_tenure");
    await until(() => text(b).includes("Working"), "pending text");
    assert.strictEqual(b.ids.send.disabled, true);                         // can't ask twice at once
    assert.ok(b.store.get("lpa.pending"), "pending job is saved");
    s.state.hold = false;
    await done;
    assert.ok(text(b).includes("VERIFIED") && text(b).includes("30 years") && text(b).includes("page 12"));
    assert.ok(text(b).includes("Report (.md)"));
    assert.strictEqual(b.store.get("lpa.pending"), undefined);              // cleared once finished
    assert.deepStrictEqual(JSON.parse(b.store.get("lpa.conversations")).map((c) => c.id), ["sess12345678"]);
    assert.ok(b.ids.convs.textContent.includes("loan_tenure"));
    assert.strictEqual(b.ids.send.disabled, false);
  });

  await test("a follow-up reuses the session id and the chosen options", async () => {
    const s = fakeServer({ slowPolls: 0 }); const b = makeBrowser({ server: s.handler });
    await until(() => text(b).includes("Ask for a field"), "load");
    await ask(b, "loan_tenure");
    s.state.jobs.chat1 = undefined; s.state.polls = 0;
    b.ids.kinds.value = "date"; b.ids.fast.checked = true; b.ids.source.value = "policy.pdf";
    await ask(b, "what about for commercial loans?");
    assert.deepStrictEqual(s.state.lastChat, { message: "what about for commercial loans?", session_id: "sess12345678", source_file: "policy.pdf", kinds: "date", fast: true });
  });

  await test("RELOAD while a question runs: it is picked up again and finishes", async () => {
    const s = fakeServer({ slowPolls: 0 }); s.state.hold = true;
    s.state.jobs.chat1 = { job_id: "chat1", status: "running", jobs_ahead: 0, running_seconds: 8, result: null, error: null };
    s.state.sessions.sess12345678 = { session_id: "sess12345678", turns: [] };
    s.state.lastChat = { message: "loan_tenure", session_id: "sess12345678" };
    const storage = { "lpa.pending": JSON.stringify({ job_id: "chat1", session_id: "sess12345678", message: "loan_tenure", started: 1 }),
      "lpa.conversations": JSON.stringify([{ id: "sess12345678", title: "loan_tenure", updated: 1 }]), "lpa.current": "sess12345678" };
    const b = makeBrowser({ server: s.handler, storage });
    await until(() => text(b).includes("Working"), "resumed pending text");
    assert.ok(text(b).includes("loan_tenure"), "your question is shown again");
    assert.strictEqual(b.ids.send.disabled, true);
    s.state.hold = false;
    await until(() => text(b).includes("VERIFIED"), "finished answer");
    assert.strictEqual(b.store.get("lpa.pending"), undefined);
    assert.strictEqual(b.ids.send.disabled, false);
    assert.strictEqual((text(b).match(/VERIFIED/g) || []).length, 1, "no duplicate answer");
  });

  await test("RELOAD after the job already finished: the answer shows once, not twice", async () => {
    const s = fakeServer();
    s.state.jobs.chat1 = { job_id: "chat1", status: "done", result: PAYLOAD };
    s.state.sessions.sess12345678 = { session_id: "sess12345678", turns: [{ turn: 1, message: "loan_tenure", payload: PAYLOAD }] };
    const b = makeBrowser({ server: s.handler, storage: { "lpa.pending": JSON.stringify({ job_id: "chat1", session_id: "sess12345678", message: "loan_tenure" }), "lpa.current": "sess12345678",
      "lpa.conversations": JSON.stringify([{ id: "sess12345678", title: "loan_tenure", updated: 1 }]) } });
    await until(() => text(b).includes("VERIFIED"), "answer");
    await new Promise((r) => setTimeout(r, 30));
    assert.strictEqual((text(b).match(/VERIFIED/g) || []).length, 1);
    assert.strictEqual(b.store.get("lpa.pending"), undefined);
  });

  await test("RELOAD when the server forgot the job: a clear message, pending cleared", async () => {
    const s = fakeServer();
    const b = makeBrowser({ server: s.handler, storage: { "lpa.pending": JSON.stringify({ job_id: "ghost", session_id: "sess12345678", message: "loan_tenure" }) } });
    await until(() => text(b).includes("no longer on the server"), "message");
    assert.strictEqual(b.store.get("lpa.pending"), undefined);
    assert.strictEqual(b.ids.send.disabled, false);
  });

  await test("a conversation the server forgot says so; deleting it removes it from the list and the server", async () => {
    const s = fakeServer();
    const b = makeBrowser({ server: s.handler, storage: { "lpa.conversations": JSON.stringify([{ id: "oldsession123", title: "old one", updated: 1 }]), "lpa.current": "oldsession123" } });
    await until(() => text(b).includes("no longer on the server"), "expired notice");
    const li = b.ids.convs.children[0]; li.children[1].onclick();
    await until(() => JSON.parse(b.store.get("lpa.conversations")).length === 0, "removed from list");
    assert.deepStrictEqual(s.state.deleted, ["oldsession123"]);   // the DELETE was sent to the server
  });

  await test("Report (.md) is built from the answer shown", async () => {
    const s = fakeServer({ slowPolls: 0 }); const b = makeBrowser({ server: s.handler });
    await until(() => text(b).includes("Ask for a field"), "load");
    await ask(b, "loan_tenure");
    const card = b.ids.log.all().find((e) => e.className === "actions");
    card.children[0].onclick();
    assert.strictEqual(b.downloads.length, 1);
    assert.ok(/^report_loan_tenure_\d{4}-\d{2}-\d{2}\.md$/.test(b.downloads[0].download), b.downloads[0].download);
    const md = b.ctx.lastBlob.text;
    for (const piece of ["# Lending policy check: loan_tenure", "- Question: loan_tenure", "| Maximum | 30 years |", "> The repayment period shall not extend beyond thirty (30) years.",
      "Source: policy.pdf, page 12", "x \\| y z", "1 extraction attempt(s)", "| extract | 1.2 |", "Spot-check the quoted page"]) assert.ok(md.includes(piece), "missing: " + piece + "\n" + md);
  });

  await test("hostile text from the server is shown as text, never turned into elements", async () => {
    const evil = { ...PAYLOAD, answer: "<img src=x onerror=alert(1)>", source_quote: "<script>alert(2)</script>", findings: [{ ...PAYLOAD.findings[0], quote: "<b>bold</b>" }] };
    const s = fakeServer({ slowPolls: 0, payload: evil }); const b = makeBrowser({ server: s.handler });
    await until(() => text(b).includes("Ask for a field"), "load");
    await ask(b, "loan_tenure");
    assert.ok(text(b).includes("<img src=x onerror=alert(1)>") && text(b).includes("<script>alert(2)</script>"));
    assert.ok(!b.ids.log.all().some((e) => ["img", "script", "b"].includes(e.tag)), "no element was created from server text");
  });

  await test("upload: shows indexed and rejected files and refreshes the list; a 400 lists why", async () => {
    const s = fakeServer(); const base = s.handler;
    const server = async (path, o = {}) => {
      if (path === "/documents?wait=false") {
        if (o.body.parts.some((p) => p[1] === "bad")) return J(400, { detail: { message: "No file was accepted.", rejected: [{ file: "n.txt", error: "only .docx, .pdf files are accepted." }] } });
        s.state.jobs.up1 = { job_id: "up1", status: "done", result: { indexed: [{ file: "new.pdf", chunks: 3 }], failed: [], rejected: [{ file: "n.txt", error: "bad type" }], documents: [{ name: "new.pdf", chunks: 3 }], total_chunks: 3 } };
        return J(202, { job_id: "up1", status: "queued" });
      }
      return base(path, o);
    };
    const b = makeBrowser({ server }); await until(() => text(b).includes("Ask for a field"), "load");
    b.ids.files.files = ["good"]; await b.ids.upload.onclick();
    assert.ok(b.ids["upload-note"].textContent.includes("✓ new.pdf: 3 chunks") && b.ids["upload-note"].textContent.includes("✗ n.txt: bad type"));
    assert.ok(b.ids.docs.textContent.includes("new.pdf"));
    b.ids.files.files = ["bad"]; await b.ids.upload.onclick();
    assert.ok(b.ids["upload-note"].textContent.includes("No file was accepted.") && b.ids["upload-note"].textContent.includes("n.txt"));
    assert.strictEqual(b.ids.upload.disabled, false);
  });

  return results;
};
