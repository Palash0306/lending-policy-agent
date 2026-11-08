// Runs every web page scenario. Exit code 1 if any fails.   node tests/ui/run_all.js
(async () => {
  const results = [...(await require("./scenarios_chat.js")()), ...(await require("./scenarios_batch.js")())];
  console.log(results.join("\n"));
  const failed = results.filter((r) => r.startsWith("FAIL")).length;
  console.log(failed ? `\n${failed} FAILED` : `\nALL PASSED (${results.length} scenarios)`);
  process.exit(failed ? 1 : 0);
})();
