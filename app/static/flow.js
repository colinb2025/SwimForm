/* Onboarding: swipe (or tap Next) through the filming steps. Without this script the steps
   still scroll sideways and the "Got it" button is always there. */
(function () {
  var root = document.querySelector("[data-flow]");
  if (!root) return;
  var track = root.querySelector(".flow-track");
  var n = track.children.length;
  var bars = root.querySelectorAll(".flow-bar i");
  var next = root.querySelector(".flow-next"), back = root.querySelector(".flow-back");
  var i = 0;
  root.classList.add("js-flow");
  next.hidden = false;

  function go(to) {
    to = Math.max(0, Math.min(n - 1, to));
    var quick = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    track.scrollTo({ left: to * track.clientWidth, behavior: quick ? "auto" : "smooth" });
  }
  function paint() {
    bars.forEach(function (b, k) { b.classList.toggle("on", k <= i); });
    root.classList.toggle("at-end", i === n - 1);
    back.hidden = i === 0;
    Array.prototype.forEach.call(track.children, function (s, k) { s.setAttribute("aria-hidden", k === i ? "false" : "true"); });
  }
  var frame;
  track.addEventListener("scroll", function () {
    cancelAnimationFrame(frame);
    frame = requestAnimationFrame(function () {
      var now = Math.round(track.scrollLeft / track.clientWidth);
      if (now !== i) { i = now; paint(); }
    });
  }, { passive: true });
  next.addEventListener("click", function () { go(i + 1); });
  back.addEventListener("click", function () { go(i - 1); });
  window.addEventListener("resize", function () { track.scrollLeft = i * track.clientWidth; });
  paint();
})();
