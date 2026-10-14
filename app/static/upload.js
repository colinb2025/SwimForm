/* Upload page enhancements. The form works without this script; it adds:
   - a preview frame of each chosen clip, so the athlete can see it's in the right slot
   - early warnings for clips that are too short, too long, or filmed vertically
   - real upload progress, then a "measuring" state while the server works */
(function () {
  var MIN_S = 12, MAX_S = 120;

  function fmtSeconds(s) {
    return s < 60 ? Math.round(s) + " seconds" : (s / 60).toFixed(1).replace(".0", "") + " minutes";
  }

  function warningsFor(video) {
    var out = [], d = video.duration;
    if (isFinite(d) && d < MIN_S) {
      out.push("This clip is " + fmtSeconds(d) + " long. We need at least 15 seconds of steady swimming to read enough strokes.");
    } else if (isFinite(d) && d > MAX_S) {
      out.push("This clip is " + fmtSeconds(d) + " long. It will work, but trimming to the 15–30 seconds of steady swimming makes it faster.");
    }
    if (video.videoHeight > video.videoWidth) {
      out.push("This clip is vertical. Turn the phone sideways so the whole swimmer fits in frame.");
    }
    return out;
  }

  function preview(slot, file) {
    var box = slot.querySelector(".preview");
    var canvas = box.querySelector("canvas");
    var meta = box.querySelector(".preview-meta");
    var warn = slot.querySelector(".warn");
    warn.hidden = true;
    warn.innerHTML = "";
    box.hidden = true;
    if (!file) return;

    var url = URL.createObjectURL(file);
    var v = document.createElement("video");
    v.muted = true;
    v.playsInline = true;
    v.preload = "metadata";
    v.src = url;

    v.addEventListener("loadedmetadata", function () {
      var w = warningsFor(v);
      if (w.length) {
        warn.innerHTML = w.map(function (t) { return "<p>" + t + "</p>"; }).join("");
        warn.hidden = false;
      }
      meta.textContent = fmtSeconds(v.duration) + (v.videoWidth ? ", " + v.videoWidth + "×" + v.videoHeight : "");
      v.currentTime = Math.min(1.5, v.duration / 3 || 0);
    });
    v.addEventListener("seeked", function () {
      var ratio = v.videoHeight / v.videoWidth || 9 / 16;
      canvas.width = 480;
      canvas.height = Math.round(480 * ratio);
      canvas.getContext("2d").drawImage(v, 0, 0, canvas.width, canvas.height);
      box.hidden = false;
      URL.revokeObjectURL(url);
    });
    v.addEventListener("error", function () {
      // Some phone formats (e.g. HEVC .mov) don't decode in every browser. The upload still works.
      meta.textContent = "No preview in this browser, but the clip will still upload.";
      canvas.width = 0;
      box.hidden = false;
      URL.revokeObjectURL(url);
    });
  }

  document.querySelectorAll(".slot").forEach(function (slot) {
    var input = slot.querySelector(".file input");
    var name = slot.querySelector(".file-name");
    input.addEventListener("change", function () {
      var f = input.files[0];
      name.textContent = f ? f.name : name.dataset.empty;
      slot.classList.toggle("chosen", !!f);
      slot.classList.remove("has-problem");
      var prob = slot.querySelector(".problem");
      if (prob) prob.remove();
      preview(slot, f);
    });
  });

  // Reports opened in this browser (stored on the device by the report page).
  try {
    var recent = JSON.parse(localStorage.getItem("swimform.recent") || "[]");
    var box = document.querySelector(".recent");
    if (recent.length && box) {
      var ul = box.querySelector(".recent-list");
      recent.forEach(function (r) {
        var li = document.createElement("li");
        var a = document.createElement("a");
        a.href = "/analyses/" + encodeURIComponent(r.id);
        a.textContent = r.date;
        li.appendChild(a);
        if (r.summary) li.appendChild(document.createTextNode(", " + r.summary));
        ul.appendChild(li);
      });
      box.hidden = false;
    }
  } catch (e) { /* storage unavailable */ }

  var form = document.getElementById("upload-form");
  form.addEventListener("submit", function (e) {
    if (!window.XMLHttpRequest || !window.FormData) return;   // plain form post
    e.preventDefault();
    var button = form.querySelector(".go");
    var progress = form.querySelector(".progress");
    var bar = progress.querySelector(".bar span");
    var label = progress.querySelector(".progress-label");
    button.disabled = true;
    button.textContent = "Uploading…";
    progress.hidden = false;
    progress.setAttribute("aria-busy", "true");

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
      label.textContent = "Checking both clips, then measuring your stroke. This takes about a minute.";
    });
    xhr.addEventListener("load", function () {
      if (xhr.status >= 200 && xhr.status < 300 && xhr.responseURL && xhr.responseURL.indexOf("/analyses/") !== -1) {
        window.location.assign(xhr.responseURL);
      } else {
        // A clip problem (422) comes back as the form with the fix written in; show it.
        document.open();
        document.write(xhr.responseText);
        document.close();
      }
    });
    xhr.addEventListener("error", function () {
      button.disabled = false;
      button.textContent = "Analyse my stroke";
      progress.classList.remove("working");
      label.textContent = "The upload stopped. Check your connection and try again.";
    });
    xhr.send(new FormData(form));
  });
})();
