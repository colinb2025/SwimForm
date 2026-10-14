/* Report page: open every collapsible section before printing, so the printout is complete. */
(function () {
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
