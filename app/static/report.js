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

  document.querySelectorAll(".print-report").forEach(function (b) {
    b.hidden = false;
    b.addEventListener("click", function () { window.print(); });
  });
})();
