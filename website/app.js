/**
 * SDX Codebase Atlas — dashboard UI, file index, graph, intel drawer.
 */
(function () {
  const areasEl = document.getElementById("areas");
  const tocNavEl = document.getElementById("toc-nav");
  const intelDrawer = document.getElementById("intel-drawer");
  const intelBody = document.getElementById("intel-body");
  const intelTitle = document.getElementById("intel-title");
  const intelClose = document.getElementById("intel-close");
  const intelScrim = document.getElementById("intel-scrim");
  const searchEl = document.getElementById("search");
  const filterTopEl = document.getElementById("filter-top");
  const filterAtlasEl = document.getElementById("filter-atlas");
  const filterRoleEl = document.getElementById("filter-role");
  const statsEl = document.getElementById("stats");
  const errEl = document.getElementById("load-error");
  const statCountEl = document.getElementById("stat-count");
  const statEdgesEl = document.getElementById("stat-edges");
  const genTimeEl = document.getElementById("gen-time");

  /** @type {Map<string, object>} */
  let byPath = new Map();

  function normalize(s) {
    return (s || "").toLowerCase();
  }

  function splitPath(p) {
    const i = p.lastIndexOf("/");
    if (i === -1) return { dir: "", base: p };
    return { dir: p.slice(0, i + 1), base: p.slice(i + 1) };
  }

  const ATLAS_TAG_LABELS = {
    image_gen: "Image gen",
    book_comic: "Book / comic",
    training: "Training",
    sampling: "Sampling",
    shared_core: "Shared core",
    vit_scorer: "ViT scorer",
    dataset: "Dataset",
    text_encode: "Text enc.",
    diffusion: "Diffusion",
    dit: "DiT",
    utilities: "Utils",
    other: "Other",
  };

  /** Short monospace label for file type (readable, no emoji). */
  function extLabel(ext) {
    const m = {
      ".py": "PY",
      ".md": "MD",
      ".toml": "TOML",
      ".json": "JSON",
      ".yaml": "YML",
      ".yml": "YML",
      ".rs": "RS",
      ".zig": "ZIG",
      ".cpp": "C++",
      ".c": "C",
      ".h": "H",
      ".hpp": "H++",
      ".go": "GO",
      ".sh": "SH",
      ".ps1": "PS1",
      ".bat": "BAT",
      ".mjs": "JS",
      ".js": "JS",
      ".txt": "TXT",
      "(none)": "—",
    };
    if (m[ext]) return m[ext];
    const e = (ext || "").replace(".", "").toUpperCase();
    return e.slice(0, 5) || "FILE";
  }

  function areaSlug(top) {
    const s = String(top || "root")
      .replace(/[^\w.-]+/g, "-")
      .replace(/^-+|-+$/g, "");
    return "area-" + (s || "root");
  }

  function escapeHtml(s) {
    const d = document.createElement("div");
    d.textContent = s;
    return d.innerHTML;
  }

  function summaryOf(f) {
    return f.summary || f.description || "—";
  }

  function detailOf(f) {
    return f.detail || f.summary || f.description || "—";
  }

  /** Richer blurb from generator (prompt→image + training); falls back to docstring line. */
  function atlasSummaryOf(f) {
    if (f.atlas_summary != null && String(f.atlas_summary).trim()) {
      return String(f.atlas_summary);
    }
    return summaryOf(f);
  }

  function searchBlob(f) {
    const im = (f.imports || []).join(" ");
    const ib = (f.imported_by || []).join(" ");
    const tags = (f.atlas_tags || []).join(" ");
    return [f.path, summaryOf(f), atlasSummaryOf(f), detailOf(f), f.ext, f.role, tags, im, ib].join(" ");
  }

  function atlasTagsHtml(tags) {
    if (!tags || !tags.length) return "";
    const items = tags
      .map((tag) => {
        const label = ATLAS_TAG_LABELS[tag] || tag.replace(/_/g, " ");
        const safe = String(tag).replace(/[^a-z0-9_-]/gi, "");
        const cls = `atlas-tag atlas-tag--${safe}`;
        return `<li class="${cls}" title="${escapeHtml(tag)}">${escapeHtml(label)}</li>`;
      })
      .join("");
    return `<ul class="atlas-tags" aria-label="Pipeline tags">${items}</ul>`;
  }

  function countEdges(files) {
    let n = 0;
    for (const f of files) {
      n += (f.imports && f.imports.length) || 0;
    }
    return n;
  }

  function setView(name) {
    const ids = ["overview", "files", "graph", "pipeline"];
    ids.forEach((n) => {
      const el = document.getElementById("view-" + n);
      if (!el) return;
      const on = n === name;
      el.hidden = !on;
      el.classList.toggle("view--active", on);
    });
    document.querySelectorAll(".nav-item[data-view]").forEach((btn) => {
      btn.classList.toggle("nav-item--active", btn.dataset.view === name);
    });
  }

  function openIntel(f) {
    if (!intelDrawer || !intelBody || !intelTitle) return;
    const base = f.path.split("/").pop() || f.path;
    intelTitle.textContent = base;
    intelBody.innerHTML = buildIntelHtml(f);
    intelDrawer.classList.add("intel-drawer--open");
    intelDrawer.setAttribute("aria-hidden", "false");
    if (intelScrim) intelScrim.hidden = false;
  }

  function closeIntel() {
    if (!intelDrawer) return;
    intelDrawer.classList.remove("intel-drawer--open");
    intelDrawer.setAttribute("aria-hidden", "true");
    if (intelScrim) intelScrim.hidden = true;
  }

  function buildIntelHtml(f) {
    const imp = f.imports || [];
    const iby = f.imported_by || [];
    const impHtml =
      imp.length === 0
        ? "<p class=\"intel-empty\">No in-repo imports resolved.</p>"
        : "<ul class=\"intel-list\">" +
          imp
            .map((p) => {
              const has = byPath.has(p);
              if (has) {
                return `<li><button type="button" class="btn-link nav-to" data-target="${encodeURIComponent(p)}">${escapeHtml(p)}</button></li>`;
              }
              return `<li><span class="muted">${escapeHtml(p)}</span></li>`;
            })
            .join("") +
          "</ul>";
    const ibyHtml =
      iby.length === 0
        ? "<p class=\"intel-empty\">Nothing in the index imports this file directly.</p>"
        : "<ul class=\"intel-list\">" +
          iby
            .map((p) => {
              const has = byPath.has(p);
              if (has) {
                return `<li><button type="button" class="btn-link nav-to" data-target="${encodeURIComponent(p)}">${escapeHtml(p)}</button></li>`;
              }
              return `<li><span>${escapeHtml(p)}</span></li>`;
            })
            .join("") +
          "</ul>";
    const tags = atlasTagsHtml(f.atlas_tags);
    return `
      <p class="intel-path">${escapeHtml(f.path)}</p>
      <div class="intel-section"><h3>Purpose</h3><p>${escapeHtml(atlasSummaryOf(f))}</p></div>
      ${tags ? `<div class="intel-section"><h3>Pipeline tags</h3>${tags}</div>` : ""}
      <div class="intel-section"><h3>Module docstring</h3><pre class="intel-pre">${escapeHtml(detailOf(f))}</pre></div>
      <div class="intel-cols">
        <div class="intel-section"><h3>Imports (in-repo)</h3>${impHtml}</div>
        <div class="intel-section"><h3>Imported by</h3>${ibyHtml}</div>
      </div>
    `;
  }

  document.querySelectorAll(".nav-item[data-view]").forEach((btn) => {
    btn.addEventListener("click", () => setView(btn.dataset.view));
  });
  document.querySelectorAll("[data-jump]").forEach((btn) => {
    btn.addEventListener("click", () => setView(btn.getAttribute("data-jump")));
  });
  if (intelClose) intelClose.addEventListener("click", closeIntel);
  if (intelScrim) intelScrim.addEventListener("click", closeIntel);

  function buildToc(visibleTops) {
    if (!tocNavEl) return;
    tocNavEl.replaceChildren();
    const overview = document.createElement("a");
    overview.href = "#pipeline-map";
    overview.textContent = "Prompt → image map";
    tocNavEl.appendChild(overview);
    for (const top of visibleTops) {
      const a = document.createElement("a");
      a.href = "#" + areaSlug(top);
      a.textContent = top;
      tocNavEl.appendChild(a);
    }
  }

  function render(data) {
    const files = data.files || [];
    byPath = new Map(files.map((f) => [f.path, f]));

    const generated = data.generated_utc || "";
    genTimeEl.textContent = generated
      ? new Date(generated).toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" })
      : "—";

    statCountEl.textContent = String(files.length);
    statEdgesEl.textContent = String(countEdges(files));

    const byTop = new Map();
    for (const f of files) {
      const t = f.top || "(root)";
      if (!byTop.has(t)) byTop.set(t, []);
      byTop.get(t).push(f);
    }

    const tops = [...byTop.keys()].sort((a, b) => {
      if (a === "(root)") return -1;
      if (b === "(root)") return 1;
      return a.localeCompare(b);
    });

    filterTopEl.innerHTML = '<option value="__all__">All areas</option>';
    for (const t of tops) {
      const opt = document.createElement("option");
      opt.value = t;
      opt.textContent = `${t} (${byTop.get(t).length})`;
      filterTopEl.appendChild(opt);
    }

    function build(filterTop, role, atlas, q) {
      const qq = normalize(q).trim();
      let shown = 0;
      const parts = [];
      let delay = 0;
      const visibleTops = [];

      for (const top of tops) {
        if (filterTop && filterTop !== "__all__" && top !== filterTop) continue;
        let group = byTop.get(top);
        if (role && role !== "__all__") {
          group = group.filter((f) => (f.role || "source") === role);
        }
        if (atlas && atlas !== "__all__") {
          group = group.filter((f) => (f.atlas_tags || []).includes(atlas));
        }
        if (qq) {
          group = group.filter((f) => normalize(searchBlob(f)).includes(qq));
        }
        if (!group.length) continue;

        visibleTops.push(top);
        shown += group.length;

        const sec = document.createElement("section");
        sec.className = "area";
        sec.id = areaSlug(top);

        const head = document.createElement("div");
        head.className = "area-head";
        head.innerHTML = `<h2><span class="area-name">${escapeHtml(top)}</span><span class="count">${group.length} file${group.length === 1 ? "" : "s"}</span></h2>`;
        sec.appendChild(head);

        const grid = document.createElement("div");
        grid.className = "grid";

        for (const f of group) {
          const card = buildCard(f, delay);
          delay += 0.015;
          grid.appendChild(card);
        }

        sec.appendChild(grid);
        parts.push(sec);
      }

      areasEl.replaceChildren(...parts);
      buildToc(visibleTops);
      statsEl.innerHTML = `Showing <strong>${shown}</strong> of <strong>${files.length}</strong> files`;
    }

    function buildCard(f, delaySec) {
      const { dir, base } = splitPath(f.path);
      const role = f.role || "source";
      const article = document.createElement("article");
      article.className = "card";
      article.dataset.role = role;
      article.dataset.path = f.path;
      article.style.setProperty("--delay", `${Math.min(delaySec, 0.9)}s`);

      const label = extLabel(f.ext || "");
      const tagsHtml = atlasTagsHtml(f.atlas_tags);

      const imp = f.imports || [];
      const iby = f.imported_by || [];

      const impHtml =
        imp.length === 0
          ? '<p class="empty">No in-repo imports resolved (stdlib / third-party only, or non-Python file).</p>'
          : "<ul>" +
            imp
              .map((p) => {
                const has = byPath.has(p);
                if (has) {
                  return `<li><button type="button" class="btn-link nav-to" data-target="${encodeURIComponent(p)}">${escapeHtml(p)}</button></li>`;
                }
                return `<li><span class="muted">${escapeHtml(p)}</span></li>`;
              })
              .join("") +
            "</ul>";

      const ibyHtml =
        iby.length === 0
          ? '<p class="empty">Nothing else in the index imports this file directly.</p>'
          : "<ul>" +
            iby
              .map((p) => {
                const has = byPath.has(p);
                if (has) {
                  return `<li><button type="button" class="btn-link nav-to" data-target="${encodeURIComponent(p)}">${escapeHtml(p)}</button></li>`;
                }
                return `<li><span>${escapeHtml(p)}</span></li>`;
              })
              .join("") +
            "</ul>";

      article.innerHTML = `
        <div class="card-inner">
          <div class="card-top">
            <div class="file-ico" title="${escapeHtml(f.ext || "")}">${escapeHtml(label)}</div>
            <div class="path-block">
              <div class="path">${dir ? `<span class="dir">${escapeHtml(dir)}</span>` : ""}${escapeHtml(base)}</div>
            </div>
            <div class="badges">
              <span class="badge role">${escapeHtml(role)}</span>
              <span class="badge ext">${escapeHtml(f.ext || "—")}</span>
            </div>
          </div>
          <p class="summary summary--atlas">${escapeHtml(atlasSummaryOf(f))}</p>
          ${tagsHtml}
          <details class="more">
            <summary>Module docstring &amp; import graph</summary>
            <div class="detail-body">${escapeHtml(detailOf(f))}</div>
            <div class="rels">
              <div>
                <h4>Imports</h4>
                ${impHtml}
              </div>
              <div>
                <h4>Imported by</h4>
                ${ibyHtml}
              </div>
            </div>
          </details>
        </div>
      `;
      return article;
    }

    function refresh() {
      const atlasVal = filterAtlasEl ? filterAtlasEl.value : "__all__";
      build(filterTopEl.value, filterRoleEl.value, atlasVal, searchEl.value);
    }

    searchEl.addEventListener("input", refresh);
    filterTopEl.addEventListener("change", refresh);
    if (filterAtlasEl) filterAtlasEl.addEventListener("change", refresh);
    filterRoleEl.addEventListener("change", refresh);

    document.addEventListener("click", (e) => {
      const btn = e.target.closest(".nav-to");
      if (btn) {
        const raw = btn.getAttribute("data-target");
        if (!raw) return;
        const target = decodeURIComponent(raw);
        const el = document.querySelector(`article.card[data-path="${CSS.escape(target)}"]`);
        if (el) {
          setView("files");
          el.scrollIntoView({ behavior: "smooth", block: "center" });
          el.classList.remove("flash");
          void el.offsetWidth;
          el.classList.add("flash");
          const det = el.querySelector("details.more");
          if (det && !det.open) det.open = true;
        }
        return;
      }
      if (e.target.closest("#intel-body")) return;
      if (e.target.closest("summary, .btn-link, a, button, details")) return;
      const card = e.target.closest("article.card");
      if (!card) return;
      const path = card.dataset.path;
      const file = byPath.get(path);
      if (file) openIntel(file);
    });

    areasEl.addEventListener("keydown", (e) => {
      if (e.key !== "Enter" && e.key !== " ") return;
      const inner = e.target.closest(".card-inner");
      if (!inner || e.target.closest("summary, .btn-link, a")) return;
      const card = e.target.closest("article.card");
      if (!card) return;
      e.preventDefault();
      const file = byPath.get(card.dataset.path);
      if (file) openIntel(file);
    });

    document.addEventListener("keydown", (e) => {
      if (e.key === "Escape") {
        closeIntel();
        return;
      }
      if (e.key === "/" && document.activeElement !== searchEl) {
        e.preventDefault();
        searchEl.focus();
      }
    });

    refresh();
  }

  function showErr() {
    errEl.hidden = false;
  }

  function hideErr() {
    errEl.hidden = true;
  }

  if (window.__SDX_CODEBASE__ && window.__SDX_CODEBASE__.files) {
    hideErr();
    render(window.__SDX_CODEBASE__);
  } else {
    fetch("files.json")
      .then((r) => {
        if (!r.ok) throw new Error(r.statusText);
        return r.json();
      })
      .then((data) => {
        hideErr();
        render(data);
      })
      .catch(() => {
        showErr();
        statsEl.textContent = "";
        areasEl.innerHTML = "";
        if (tocNavEl) tocNavEl.replaceChildren();
        statCountEl.textContent = "0";
        statEdgesEl.textContent = "0";
      });
  }
})();
