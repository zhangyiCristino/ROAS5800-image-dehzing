"use strict";

const comparison = document.querySelector("#comparison");
const divider = document.querySelector("#divider");
const mode = document.querySelector("#compare-mode");
const sampleList = document.querySelector("#sample-list");
const sampleInfo = document.querySelector("#sample-info");
const status = document.querySelector("#demo-status");
const scoreBody = document.querySelector("#scene-scores");
const leftImage = document.querySelector("#left-image");
const rightImage = document.querySelector("#right-image");
const outputLink = document.querySelector("#download-output");
const labels = {
  input: "Hazy input",
  base: "Frozen PGL-Net-T",
  reference: "Clean reference",
  refiner: "Side-Refiner",
};
let activeSample;
let renderRevision = 0;
const sampleButtons = new Map();

function imagePath(id, name) {
  if (name === "input" || name === "reference") {
    return "../assignment2/samples/" + id + "/" + name + ".png";
  }
  return "samples/" + id + "/" + name + ".webp";
}

function setDivider() {
  comparison.style.setProperty("--reveal", divider.value + "%");
}

function loadImage(path) {
  const candidate = new Image();
  candidate.src = path;
  return candidate.decode().then(() => candidate.src);
}

async function renderComparison() {
  if (!activeSample) return;
  const revision = ++renderRevision;
  const left = mode.value;
  const sample = activeSample;
  status.hidden = true;
  try {
    const [leftSrc, rightSrc] = await Promise.all([
      loadImage(imagePath(sample.id, left)),
      loadImage(imagePath(sample.id, "refiner")),
    ]);
    if (revision !== renderRevision) return;
    leftImage.src = leftSrc;
    leftImage.alt = labels[left];
    rightImage.src = rightSrc;
    document.querySelector("#left-label").textContent = labels[left];
    sampleInfo.textContent = "RW²AH · " + sample.id + " · " + sample.width + " × " + sample.height;
    outputLink.href = rightSrc;
    outputLink.download = sample.id + "-side-refiner.webp";
    for (const [id, button] of sampleButtons) {
      button.setAttribute("aria-pressed", String(id === sample.id));
    }
    const rows = ["input", "base", "refiner"].map((key) => {
      const row = document.createElement("tr");
      const name = document.createElement("th");
      name.scope = "row";
      name.textContent = key === "base" ? "Frozen base" : labels[key];
      row.append(name);
      for (const metric of ["psnr", "ssim"]) {
        const value = document.createElement("td");
        value.textContent = sample.metrics[key][metric].toFixed(4);
        row.append(value);
      }
      return row;
    });
    scoreBody.replaceChildren(...rows);
  } catch {
    if (revision !== renderRevision) return;
    status.textContent = "This comparison could not be loaded. Select a different scene or refresh.";
    status.hidden = false;
  }
}

async function loadSamples() {
  try {
    const response = await fetch("samples.json");
    if (!response.ok) throw new Error("Sample list unavailable");
    const samples = await response.json();
    for (const sample of samples) {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "sample";
      button.setAttribute("aria-pressed", "false");
      const thumbnail = new Image();
      thumbnail.src = imagePath(sample.id, "input");
      thumbnail.alt = "";
      thumbnail.width = sample.width;
      thumbnail.height = sample.height;
      const name = document.createElement("span");
      name.textContent = sample.id;
      button.append(thumbnail, name);
      button.addEventListener("click", () => {
        activeSample = sample;
        renderComparison();
      });
      sampleButtons.set(sample.id, button);
      sampleList.append(button);
    }
    activeSample = samples[0];
    renderComparison();
  } catch {
    status.textContent = "The example list could not be loaded.";
    status.hidden = false;
  }
}

divider.addEventListener("input", setDivider);
mode.addEventListener("change", renderComparison);
document.querySelector("#copy-command").addEventListener("click", async () => {
  const copyStatus = document.querySelector("#copy-status");
  try {
    await navigator.clipboard.writeText(document.querySelector("#demo-command").textContent);
    copyStatus.textContent = "Command copied.";
  } catch {
    copyStatus.textContent = "Select the command to copy it.";
  }
});
setDivider();
loadSamples();
