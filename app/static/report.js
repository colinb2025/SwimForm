/* Report page:
   - remember this report in this browser, so the upload page can list "your recent reports"
     (kept on the device only; there are no accounts yet, so there is no shared list)
   - open every collapsible section before printing, so the printout is complete */
(function () {
  var el = document.querySelector("[data-report-id]");
  if (el) {
    try {
      var key = "swimform.recent";
      var list = JSON.parse(localStorage.getItem(key) || "[]").filter(function (r) { return r.id !== el.dataset.reportId; });
      list.unshift({ id: el.dataset.reportId, date: el.dataset.reportDate, summary: el.dataset.reportSummary });
      localStorage.setItem(key, JSON.stringify(list.slice(0, 8)));
    } catch (e) { /* storage unavailable: nothing to remember */ }
  }

  var opened = [];
  window.addEventListener("beforeprint", function () {
    document.querySelectorAll("details:not([open])").forEach(function (d) { d.open = true; opened.push(d); });
  });
  window.addEventListener("afterprint", function () {
    opened.forEach(function (d) { d.open = false; });
    opened = [];
  });

  var printBtn = document.querySelector(".print-report");
  if (printBtn) {
    printBtn.hidden = false;
    printBtn.addEventListener("click", function () { window.print(); });
  }
})();
