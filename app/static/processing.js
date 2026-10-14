/* Wait for the background analysis, then show whatever it produced (report, or what to fix). */
(function () {
  var box = document.querySelector(".job");
  if (!box || !window.fetch) return;
  var url = box.getAttribute("data-status-url");
  var delay = 1500;
  function poll() {
    fetch(url, { credentials: "same-origin", headers: { Accept: "application/json" } })
      .then(function (r) { return r.ok ? r.json() : null; })
      .then(function (d) {
        if (d && d.status !== "queued" && d.status !== "running") { window.location.reload(); return; }
        if (d && d.status === "running") {
          var steps = box.querySelectorAll(".job-steps li");
          if (steps[1] && !steps[1].classList.contains("active")) {
            steps[1].classList.replace("next", "active");
            box.querySelector(".progress-label").textContent = "Working on it.";
          }
        }
        delay = Math.min(delay * 1.3, 6000);
        setTimeout(poll, delay);
      })
      .catch(function () { delay = Math.min(delay * 2, 15000); setTimeout(poll, delay); });
  }
  setTimeout(poll, delay);
})();
