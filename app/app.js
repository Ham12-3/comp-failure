// Front end for the Company Failure Predictor API.
// Every value from the API is inserted with textContent (never innerHTML).
"use strict";

const $ = (id) => document.getElementById(id);
const COMPANY_NUMBER = /^(?=.*\d)[A-Za-z0-9]{6,8}$/;
const STATUS_LABELS = {
  active: "Active",
  dissolved: "Dissolved",
  liquidation: "In liquidation",
  administration: "In administration",
  receivership: "Receiver appointed",
  "voluntary-arrangement": "Voluntary arrangement",
};
const BAND_LABELS = { typical: "Typical risk", elevated: "Elevated risk", high: "High risk" };

let searchTimer = null;

function setStatus(text) {
  $("status").textContent = text || "";
}

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined && text !== null) node.textContent = text;
  return node;
}

function formatDate(iso) {
  if (!iso) return "";
  return new Date(iso + "T00:00:00").toLocaleDateString("en-GB", {
    day: "numeric", month: "long", year: "numeric",
  });
}

async function search(query) {
  const results = $("results");
  try {
    const response = await fetch(`/search?q=${encodeURIComponent(query)}`);
    if (!response.ok) throw new Error((await response.json()).detail || response.statusText);
    const { items } = await response.json();
    results.replaceChildren();
    for (const item of items) {
      const li = el("li");
      const button = el("button", "result");
      button.type = "button";
      button.append(el("span", "result-name", item.title || item.company_number));
      const meta = [item.company_number, STATUS_LABELS[item.company_status] || item.company_status,
                    item.address_snippet].filter(Boolean).join(" · ");
      button.append(el("span", "result-meta", meta));
      button.addEventListener("click", () => predict(item.company_number));
      li.append(button);
      results.append(li);
    }
    results.hidden = items.length === 0;
    setStatus(items.length ? "" : "No companies found.");
  } catch (error) {
    results.hidden = true;
    setStatus(`Search failed: ${error.message}`);
  }
}

async function predict(companyNumber) {
  $("results").hidden = true;
  $("card").hidden = true;
  setStatus(`Scoring ${companyNumber}: fetching records and the latest accounts…`);
  try {
    const response = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ company_number: companyNumber }),
    });
    const body = await response.json();
    if (!response.ok) throw new Error(body.detail || response.statusText);
    render(body);
    setStatus("");
  } catch (error) {
    setStatus(`Could not score ${companyNumber}: ${error.message}`);
  }
}

function render(data) {
  $("company-name").textContent = data.company_name || data.company_number;
  const status = STATUS_LABELS[data.company_status] || data.company_status || "";
  $("company-meta").textContent = [`Company ${data.company_number}`, status].filter(Boolean).join(" · ");

  const band = $("band");
  band.hidden = !data.scored;
  band.className = `band band-${data.risk_band || ""}`;
  band.textContent = BAND_LABELS[data.risk_band] || "";

  $("message").hidden = !data.message;
  $("message").textContent = data.message || "";

  $("score").hidden = !data.scored;
  $("reasons-block").hidden = !data.scored;
  if (data.scored) {
    const pct = Math.round(data.percentile);
    $("percentile").textContent = pct;
    $("gauge-fill").setAttribute("stroke-dasharray", `${pct} 100`);
    $("gauge-fill").setAttribute("class", `gauge-fill fill-${data.risk_band}`);
    $("gauge-caption").textContent = `Riskier than ${pct}% of companies in the reference years`;
    $("probability").textContent = `${data.score_percent.toFixed(2)}%`;
    $("window").textContent =
      `Within 12 months of the accounts filed on ${formatDate(data.accounts_filed_on)} ` +
      `(period ending ${formatDate(data.balance_sheet_date)}), i.e. by ${formatDate(data.horizon_end)}.`;
    $("relative").textContent = `${data.relative_risk.toFixed(1)}× the average company's rate.`;

    const reasons = $("reasons");
    reasons.replaceChildren();
    for (const reason of data.reasons) {
      const li = el("li", `reason reason-${reason.effect}`);
      li.append(el("span", "reason-icon", reason.effect === "raises" ? "▲" : "▼"));
      li.append(el("span", "reason-text", reason.text));
      reasons.append(li);
    }
  }

  const warnings = $("warnings");
  warnings.replaceChildren(...(data.warnings || []).map((w) => el("li", null, w)));
  warnings.hidden = !(data.warnings || []).length;

  const links = $("links");
  links.replaceChildren();
  for (const link of data.links) {
    const li = el("li");
    const a = el("a", null, link.label);
    a.href = link.url;
    a.target = "_blank";
    a.rel = "noopener noreferrer";
    li.append(a);
    links.append(li);
  }
  $("model-info").textContent = data.model_trained_at
    ? `Model trained ${formatDate(data.model_trained_at.slice(0, 10))}.`
    : "";
  $("card").hidden = false;
}

$("search-form").addEventListener("submit", (event) => {
  event.preventDefault();
  const query = $("q").value.trim();
  if (COMPANY_NUMBER.test(query)) predict(query.toUpperCase());
  else if (query.length >= 2) search(query);
});

$("q").addEventListener("input", () => {
  clearTimeout(searchTimer);
  const query = $("q").value.trim();
  if (query.length < 3 || COMPANY_NUMBER.test(query)) return;
  searchTimer = setTimeout(() => search(query), 350);
});
