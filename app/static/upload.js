/* Upload page enhancements. The form works without this script; it adds:
   - drag and drop onto each slot
   - a preview frame of each chosen clip, so the athlete can see it's in the right slot
   - early warnings for clips that are too short, too long, or filmed vertically
   - real upload progress, then a "measuring" state while the server works */
(function () {
  var MIN_S = 5, MAX_S = 120;

  function fmtSeconds(s) {
    return s < 60 ? Math.round(s) + " s" : (s / 60).toFixed(1).replace(".0", "") + " min";
  }

  function warningsFor(video) {
    var out = [], d = video.duration;
    if (isFinite(d) && d < MIN_S) {
      out.push("This clip is only " + fmtSeconds(d) + ". We need at least three full strokes, so it may be too short.");
    } else if (isFinite(d) && d > MAX_S) {
      out.push("This clip is " + fmtSeconds(d) + ". It will work, but trimming it to one 50 of steady swimming is faster.");
    }
    if (video.videoHeight > video.videoWidth) {
      out.push("This clip is vertical. Turn the phone sideways so the whole swimmer fits.");
    }
    return out;
  }

  function show(slot, file) {
    var card = slot.closest(".slot");
    var name = slot.querySelector(".chosen-name");
    var info = slot.querySelector(".chosen-info");
    var canvas = slot.querySelector(".chosen-view canvas");
    var warn = slot.querySelector(".warn");
    warn.hidden = true;
    warn.innerHTML = "";
    slot.classList.toggle("chosen", !!file);
    slot.classList.remove("no-preview");
    card.classList.remove("has-problem");
    var prob = card.querySelector(".problem");
    if (prob) prob.remove();
    if (!file) return;
    name.textContent = file.name;
    info.textContent = "";

    var url = URL.createObjectURL(file);
    var v = document.createElement("video");
    v.muted = true; v.playsInline = true; v.preload = "metadata"; v.src = url;
    v.addEventListener("loadedmetadata", function () {
      var w = warningsFor(v);
      if (w.length) {
        warn.innerHTML = "<div>" + w.map(function (t) { return "<p>" + t + "</p>"; }).join("") + "</div>";
        warn.hidden = false;
      }
      info.textContent = fmtSeconds(v.duration);
      v.currentTime = Math.min(1.5, v.duration / 3 || 0);
    });
    v.addEventListener("seeked", function () {
      var ratio = v.videoHeight / v.videoWidth || 9 / 16;
      canvas.width = 480;
      canvas.height = Math.round(480 * ratio);
      canvas.getContext("2d").drawImage(v, 0, 0, canvas.width, canvas.height);
      URL.revokeObjectURL(url);
    });
    v.addEventListener("error", function () {
      // Some phone formats (e.g. HEVC .mov) don't decode in every browser; the upload still works.
      slot.classList.add("no-preview");
      info.textContent = "";
      URL.revokeObjectURL(url);
    });
  }

  document.querySelectorAll(".zone").forEach(function (slot) {
    var input = slot.querySelector(".drop input");
    var drop = slot.querySelector(".drop");
    input.addEventListener("change", function () { show(slot, input.files[0]); });
    ["dragenter", "dragover"].forEach(function (ev) {
      drop.addEventListener(ev, function (e) { e.preventDefault(); drop.classList.add("dragging"); });
    });
    ["dragleave", "drop"].forEach(function (ev) {
      drop.addEventListener(ev, function () { drop.classList.remove("dragging"); });
    });
    drop.addEventListener("drop", function (e) {
      e.preventDefault();
      if (!e.dataTransfer || !e.dataTransfer.files.length) return;
      try {
        var dt = new DataTransfer();
        dt.items.add(e.dataTransfer.files[0]);
        input.files = dt.files;
      } catch (err) { return; }
      show(slot, input.files[0]);
    });
  });

  var form = document.getElementById("upload-form");
  form.addEventListener("submit", function (e) {
    // One clip of each view is the minimum; say so here instead of after a long upload.
    var missing = null;
    form.querySelectorAll(".slot").forEach(function (card) {
      var any = Array.prototype.some.call(card.querySelectorAll("input[type=file]"), function (i) { return i.files.length; });
      var old = card.querySelector(".problem");
      if (old) old.remove();
      card.classList.toggle("has-problem", !any);
      if (any) return;
      var box = document.createElement("div");
      box.className = "notice notice-error problem";
      box.setAttribute("role", "alert");
      box.innerHTML = "<div><p>Add a " + card.getAttribute("data-view") + " clip: above water, underwater, or both.</p></div>";
      card.querySelector(".zones").before(box);
      missing = missing || card;
    });
    if (missing) { e.preventDefault(); missing.scrollIntoView({ behavior: "smooth", block: "center" }); return; }
    if (!window.XMLHttpRequest || !window.FormData) return;   // plain form post
    e.preventDefault();
    var button = form.querySelector("button[type=submit]");
    var progress = form.querySelector(".progress");
    var bar = progress.querySelector(".bar span");
    var label = progress.querySelector(".progress-label");
    var label0 = button.textContent;
    button.disabled = true;
    button.textContent = "Uploading…";
    progress.hidden = false;

    var xhr = new XMLHttpRequest();
    xhr.open("POST", form.action);
    xhr.upload.addEventListener("progress", function (ev) {
      if (!ev.lengthComputable) return;
      var pct = Math.round((100 * ev.loaded) / ev.total);
      bar.style.width = pct + "%";
      label.textContent = "Uploading your clips: " + pct + "%";
    });
    xhr.upload.addEventListener("load", function () {
      bar.style.width = "100%";
      progress.classList.add("working");
      button.textContent = "Analysing…";
      label.textContent = "Checking your clips, then measuring your stroke. This takes about a minute.";
    });
    xhr.addEventListener("load", function () {
      if (xhr.status >= 200 && xhr.status < 300 && xhr.responseURL && xhr.responseURL.indexOf("/analyses/") !== -1) {
        window.location.assign(xhr.responseURL);
      } else {
        // A clip problem (422) comes back as the form with the fix written in; show it.
        document.open(); document.write(xhr.responseText); document.close();
      }
    });
    xhr.addEventListener("error", function () {
      button.disabled = false;
      button.textContent = label0;
      progress.classList.remove("working");
      label.textContent = "The upload stopped. Check your connection and try again.";
    });
    xhr.send(new FormData(form));
  });
})();
