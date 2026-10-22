'use strict';

// This page displays original saved examples and local uploads. It does not run models.
(() => {
  const METHODS = ['PGL-Net-T', 'HNDiff-FocalNet', 'PixRestore-S'];
  const MAX_FILE_BYTES = 32 * 1024 * 1024;
  const $ = (id) => document.getElementById(id);
  const state = {hazy: null, reference: null, sample: null};
  const pending = {hazy: false, reference: false};
  const versions = {hazy: 0, reference: 0};
  const activeUrls = new Set();
  const methodInputs = [...document.querySelectorAll('input[name="method"]')];
  const cards = [...document.querySelectorAll('article[data-method]')];
  let availableSamples = [];
  let sourceInfo = null;
  let sampleVersion = 0;
  let interactionVersion = 0;
  let pendingSample = false;

  function revoke(url) {
    if (url && activeUrls.delete(url)) URL.revokeObjectURL(url);
  }

  function message(text, error = false) {
    $('status-message').textContent = text;
    $('status-message').classList.toggle('is-error', error);
    $('status-message').dataset.tone = error ? 'error' : 'info';
  }

  function selectedMethods() {
    return METHODS.filter((method) => methodInputs.some((input) => input.value === method && input.checked));
  }

  function dimensionsMismatch() {
    return state.hazy && state.reference &&
      (state.hazy.width !== state.reference.width || state.hazy.height !== state.reference.height);
  }

  function processedSize() {
    if (!state.hazy) return null;
    if (state.sample) return {width: state.sample.width, height: state.sample.height};
    const maximum = Number($('max-side').value);
    const scale = Math.min(1, maximum / Math.max(state.hazy.width, state.hazy.height));
    const round = (value) => {
      const lower = Math.floor(value);
      return value - lower === 0.5 ? lower + lower % 2 : Math.round(value);
    };
    return {width: Math.max(1, round(state.hazy.width * scale)),
      height: Math.max(1, round(state.hazy.height * scale))};
  }

  function sizeText(image) {
    return `${image.width} × ${image.height}`;
  }

  function numberOrDash(value, digits) {
    return typeof value === 'number' && Number.isFinite(value) ? value.toFixed(digits) : '—';
  }

  function updateDivider(card) {
    const range = card.querySelector('.comparison-range');
    const value = Math.max(0, Math.min(100, Number(range.value)));
    card.querySelector('.output-body').style.setProperty('--divider', `${value}%`);
    card.querySelector('.comparison-input').style.clipPath = `inset(0 ${100 - value}% 0 0)`;
    const saved = state.sample && selectedMethods().includes(card.dataset.method);
    range.setAttribute('aria-valuetext', saved
      ? `${value}% input; ${100 - value}% original saved output`
      : `${value}% input; output is not available`);
  }

  function updateSampleButtons() {
    for (const button of $('sample-list').querySelectorAll('button[data-sample-id]')) {
      const active = Boolean(state.sample && state.sample.id === button.dataset.sampleId);
      button.classList.toggle('is-active', active);
      button.setAttribute('aria-pressed', String(active));
    }
  }

  function render() {
    const selected = selectedMethods();
    const mismatch = dimensionsMismatch();
    const size = processedSize();
    for (const kind of ['hazy', 'reference']) {
      const image = state[kind];
      const preview = $(`${kind}-preview`);
      preview.hidden = !image;
      $(`${kind}-placeholder`).hidden = Boolean(image);
      $(`clear-${kind}`).hidden = !image;
      if (image) {
        const name = image.file ? image.file.name : image.name;
        preview.src = image.url;
        preview.alt = `${kind === 'hazy' ? 'Hazy image' : 'Clean reference'}: ${name}`;
        const origin = image.origin === 'saved' ? 'Original saved image' : `${(image.file.size / 1024 / 1024).toFixed(2)} MB`;
        $(`${kind}-meta`).textContent = `${name} · ${sizeText(image)} px · ${origin}`;
      } else {
        preview.removeAttribute('src');
        $(`${kind}-meta`).textContent = kind === 'hazy' ? 'No image selected' : 'No reference selected';
      }
      $(`${kind}-file`).setAttribute('aria-invalid', String(Boolean(mismatch)));
    }
    $('reference-warning').hidden = !mismatch;
    $('reference-warning').textContent = mismatch
      ? `Reference dimensions do not match: hazy ${sizeText(state.hazy)} px; reference ${sizeText(state.reference)} px. Replace or remove the reference to continue.`
      : '';
    $('selection-count').textContent = `${selected.length} selected`;
    $('max-side').disabled = Boolean(state.sample) || pendingSample;
    if (state.sample) $('max-side').value = '512';
    $('compare-button').textContent = state.sample ? 'Show saved results' : 'Compare methods';
    $('compare-button').disabled = !state.hazy || !selected.length || Boolean(mismatch) || pending.hazy || pending.reference || pendingSample;
    $('download-button').disabled = !state.sample || !selected.length || pendingSample || pending.hazy || pending.reference;
    $('download-button').title = state.sample
      ? 'Download the original saved ZIP containing input, reference, all three method outputs and recorded metrics.'
      : 'A saved comparison is required. New uploads have no model outputs to download.';
    $('download-note').textContent = state.sample
      ? 'Original saved ZIP · includes all three methods, regardless of the current selection.'
      : 'Downloads become available for a saved example. New uploads require a connected model.';

    for (const card of cards) {
      const enabled = selected.includes(card.dataset.method);
      const before = card.querySelector('.before-image');
      const after = card.querySelector('.after-image');
      const showInput = enabled && Boolean(state.hazy);
      const showOutput = enabled && Boolean(state.sample);
      card.querySelector('.input-placeholder').hidden = showInput;
      card.querySelector('.comparison-empty').hidden = showOutput;
      card.classList.toggle('is-disabled', !enabled);
      card.classList.toggle('has-saved-result', showOutput);
      before.hidden = !showInput;
      after.hidden = !showOutput;
      if (showInput) {
        before.src = state.hazy.url;
        before.alt = state.sample ? `Original saved hazy input ${state.sample.id}` : 'Uploaded hazy input; no restored output has been generated';
      } else before.removeAttribute('src');
      if (showOutput) {
        after.src = state.sample.methods[card.dataset.method].output;
        after.alt = `Original saved ${card.dataset.method} output for ${state.sample.id}`;
      } else after.removeAttribute('src');
      card.querySelector('.comparison-range').disabled = !showInput;
      card.querySelector('.output-status').textContent = !enabled ? 'Not selected'
        : showOutput ? 'Saved result' : state.hazy ? 'Waiting for model' : 'Waiting for input';
      updateDivider(card);
    }

    const body = $('metrics-body');
    body.replaceChildren();
    for (const method of selected) {
      const record = state.sample?.methods[method];
      const values = [method, size ? `${sizeText(size)} px` : '—',
        numberOrDash(record?.gpu_ms, 2), numberOrDash(record?.peak_allocated_mib, 1),
        numberOrDash(record?.psnr, 3), numberOrDash(record?.ssim, 4)];
      const row = document.createElement('tr');
      for (const text of values) {
        const cell = document.createElement('td');
        cell.textContent = text;
        row.append(cell);
      }
      body.append(row);
    }
    if (!selected.length) {
      const row = document.createElement('tr');
      const cell = document.createElement('td');
      cell.colSpan = 6;
      cell.textContent = 'Select at least one method to prepare a comparison.';
      row.append(cell);
      body.append(row);
    }
    updateSampleButtons();
  }

  function inputStatus() {
    if (pendingSample) return;
    if (dimensionsMismatch()) {
      message('The clean reference must have the same dimensions as the hazy image. Replace or remove it to continue.', true);
    } else if (!state.hazy) {
      message(state.reference ? 'Clean reference loaded. Upload a hazy image to prepare a comparison.' : 'Choose a saved example or upload a hazy image.');
    } else if (!selectedMethods().length) {
      message('Select at least one method to prepare a comparison.');
    } else if (state.sample) {
      message(`Showing the original saved run for ${state.sample.label} (${state.sample.id}). Images and metrics are recorded results, not fresh inference.`);
    } else {
      message('Input ready. Model inference is not connected in this UI preview; restored images and measured metrics are not available for new uploads.');
    }
  }

  function cancelSampleLoad() {
    interactionVersion += 1;
    sampleVersion += 1;
    pendingSample = false;
  }

  function clearSavedResults() {
    state.sample = null;
    if (availableSamples.length) $('sample-note').textContent = 'Choose a saved example, or upload your own image. Saved results are never applied to new uploads.';
  }

  function supportedImage(bytes) {
    const png = [137, 80, 78, 71, 13, 10, 26, 10];
    return png.every((value, index) => bytes[index] === value) ||
      (bytes[0] === 255 && bytes[1] === 216 && bytes[2] === 255) ||
      (bytes[0] === 82 && bytes[1] === 73 && bytes[2] === 70 && bytes[3] === 70 &&
       bytes[8] === 87 && bytes[9] === 69 && bytes[10] === 66 && bytes[11] === 80);
  }

  async function loadFile(kind, file) {
    cancelSampleLoad();
    clearSavedResults();
    const version = ++versions[kind];
    const label = kind === 'hazy' ? 'Hazy image' : 'Clean reference';
    let url = null;
    pending[kind] = true;
    render();
    message(`Checking ${label.toLowerCase()}…`);
    try {
      if (!file || !file.size) throw new Error('Choose a non-empty PNG, JPG or WebP image.');
      if (file.size > MAX_FILE_BYTES) throw new Error('The image exceeds the 32 MB file limit. Choose a smaller file.');
      const header = new Uint8Array(await file.slice(0, 12).arrayBuffer());
      if (!supportedImage(header)) throw new Error('Unsupported image. Choose a real PNG, JPG or WebP file.');
      if (version !== versions[kind]) return;
      url = URL.createObjectURL(file);
      activeUrls.add(url);
      const probe = new Image();
      probe.src = url;
      try { await probe.decode(); }
      catch { throw new Error('This image could not be opened. The file may be damaged or invalid.'); }
      if (version !== versions[kind]) return;
      if (!probe.naturalWidth || !probe.naturalHeight) throw new Error('This image has no valid dimensions.');
      const previous = state[kind];
      state[kind] = {file, url, width: probe.naturalWidth, height: probe.naturalHeight, origin: 'upload'};
      url = null;
      render();
      if (previous) revoke(previous.url);
      inputStatus();
    } catch (error) {
      if (version === versions[kind]) {
        const retained = state[kind] ? ` The previous ${label.toLowerCase()} is unchanged; its saved results have been cleared.` : '';
        message(`${error.message}${retained}`, true);
      }
    } finally {
      revoke(url);
      if (version === versions[kind]) {
        pending[kind] = false;
        $(`${kind}-file`).value = '';
        render();
      }
    }
  }

  function clearImage(kind, announce = true) {
    cancelSampleLoad();
    clearSavedResults();
    versions[kind] += 1;
    pending[kind] = false;
    const previous = state[kind];
    state[kind] = null;
    $(`${kind}-file`).value = '';
    render();
    if (previous) revoke(previous.url);
    if (announce) inputStatus();
  }

  function safeSampleUrl(value, extension) {
    if (typeof value !== 'string' || !value.startsWith('samples/') ||
        !/^[A-Za-z0-9_./-]+$/.test(value) || value.split('/').some((part) => !part || part === '.' || part === '..') ||
        !value.endsWith(extension)) throw new Error('The saved-example manifest contains an invalid local asset path.');
    return value;
  }

  function validateManifest(manifest) {
    if (manifest.schema_version !== 1 || !Array.isArray(manifest.samples) || !manifest.samples.length ||
        manifest.source?.protocol?.max_side !== 512) throw new Error('The saved-example manifest is invalid.');
    const ids = new Set();
    for (const sample of manifest.samples) {
      if (typeof sample.id !== 'string' || !sample.id || ids.has(sample.id) || typeof sample.label !== 'string' ||
          !Number.isInteger(sample.width) || sample.width < 1 || !Number.isInteger(sample.height) || sample.height < 1) {
        throw new Error('The saved-example manifest has inconsistent image information.');
      }
      ids.add(sample.id);
      safeSampleUrl(sample.input, '.png');
      safeSampleUrl(sample.reference, '.png');
      safeSampleUrl(sample.download, '.zip');
      for (const method of METHODS) {
        const record = sample.methods?.[method];
        if (!record) throw new Error('A saved example is missing a method.');
        safeSampleUrl(record.output, '.png');
        for (const key of ['psnr', 'ssim', 'gpu_ms', 'peak_allocated_mib']) {
          if (record[key] != null && (typeof record[key] !== 'number' || !Number.isFinite(record[key]))) {
            throw new Error('A saved example has an invalid recorded metric.');
          }
        }
      }
    }
    return manifest;
  }

  async function decodeSavedImage(url, sample) {
    const image = new Image();
    image.src = url;
    await image.decode();
    if (image.naturalWidth !== sample.width || image.naturalHeight !== sample.height) {
      throw new Error('Saved image dimensions do not match the recorded example.');
    }
  }

  async function loadSample(sample, isDefault = false) {
    if (!isDefault) interactionVersion += 1;
    const interaction = interactionVersion;
    const version = ++sampleVersion;
    pendingSample = true;
    const previous = [state.hazy, state.reference];
    for (const kind of ['hazy', 'reference']) {
      versions[kind] += 1;
      pending[kind] = false;
      state[kind] = null;
      $(`${kind}-file`).value = '';
    }
    state.sample = null;
    render();
    previous.forEach((image) => { if (image) revoke(image.url); });
    $('sample-note').textContent = `Loading ${sample.label} · ${sample.id}…`;
    message(`Loading the original saved images and results for ${sample.label}…`);
    try {
      // Commit all five pictures and the metrics together; partial results stay hidden.
      await Promise.all([sample.input, sample.reference, ...METHODS.map((method) => sample.methods[method].output)]
        .map((url) => decodeSavedImage(url, sample)));
      if (version !== sampleVersion || (isDefault && interaction !== interactionVersion)) return;
      state.hazy = {url: sample.input, width: sample.width, height: sample.height,
        name: `${sample.id} · hazy input`, origin: 'saved'};
      state.reference = {url: sample.reference, width: sample.width, height: sample.height,
        name: `${sample.id} · clean reference`, origin: 'saved'};
      state.sample = sample;
      $('max-side').value = '512';
      for (const card of cards) card.querySelector('.comparison-range').value = '50';
      const recorded = sourceInfo?.recorded_at_utc ? ` · recorded ${sourceInfo.recorded_at_utc.slice(0, 10)}` : '';
      $('sample-note').textContent = `${sample.label} · ${sample.id}${recorded} · original saved run at maximum side 512 px.`;
    } catch {
      if (version === sampleVersion) {
        $('sample-note').textContent = `Could not load ${sample.label}. Choose another saved example or upload your own image.`;
        message('The saved example could not be loaded completely. No partial outputs or metrics are displayed. Try another example.', true);
      }
    } finally {
      if (version === sampleVersion) {
        pendingSample = false;
        render();
        if (state.sample) inputStatus();
      }
    }
  }

  function renderSampleList() {
    $('sample-title').textContent = `Real test samples (${availableSamples.length})`;
    $('sample-list').replaceChildren();
    for (const sample of availableSamples) {
      const button = document.createElement('button');
      button.type = 'button';
      button.className = 'sample-button';
      button.dataset.sampleId = sample.id;
      button.setAttribute('aria-pressed', 'false');
      button.setAttribute('aria-label', `Load ${sample.label}, saved test image ${sample.id}`);
      const thumb = document.createElement('img');
      thumb.className = 'sample-thumb';
      thumb.src = sample.input;
      thumb.alt = '';
      const text = document.createElement('div');
      const name = document.createElement('span');
      name.className = 'sample-name';
      name.textContent = sample.label;
      const id = document.createElement('span');
      id.className = 'sample-id';
      id.textContent = sample.id;
      text.append(name, id);
      button.append(thumb, text);
      button.addEventListener('click', () => loadSample(sample));
      $('sample-list').append(button);
    }
    updateSampleButtons();
  }

  async function loadManifest() {
    $('sample-note').textContent = 'Loading saved examples…';
    try {
      const response = await fetch('samples/manifest.json', {cache: 'no-cache'});
      if (!response.ok) throw new Error('Saved examples unavailable.');
      const manifest = validateManifest(await response.json());
      sourceInfo = manifest.source;
      availableSamples = manifest.samples;
      renderSampleList();
      if (interactionVersion === 0) await loadSample(availableSamples[0], true);
      else if (!state.sample && !pendingSample) $('sample-note').textContent = 'Choose a saved example, or upload your own image.';
    } catch {
      $('sample-note').textContent = 'Saved examples are unavailable. You can still upload your own image.';
      if (interactionVersion === 0) message('Saved examples could not be loaded. Upload an image to explore the UI preview.', true);
    }
  }

  for (const kind of ['hazy', 'reference']) {
    const input = $(`${kind}-file`);
    const zone = $(`${kind}-dropzone`);
    input.addEventListener('click', () => {
      cancelSampleLoad();
      if (!state.sample && availableSamples.length) $('sample-note').textContent = 'Choose a saved example, or upload your own image.';
      render();
      inputStatus();
    });
    input.addEventListener('change', () => {
      if (input.files.length === 1) loadFile(kind, input.files[0]);
    });
    zone.addEventListener('click', (event) => {
      if (event.target.closest('button, input, a')) return;
      event.preventDefault();
      input.click();
    });
    zone.addEventListener('keydown', (event) => {
      if (event.target !== zone || !['Enter', ' '].includes(event.key)) return;
      event.preventDefault();
      input.click();
    });
    for (const eventName of ['dragenter', 'dragover']) {
      zone.addEventListener(eventName, (event) => {
        event.preventDefault();
        event.stopPropagation();
        zone.classList.add('is-dragover');
      });
    }
    zone.addEventListener('dragleave', (event) => {
      if (!zone.contains(event.relatedTarget)) zone.classList.remove('is-dragover');
    });
    zone.addEventListener('drop', (event) => {
      event.preventDefault();
      event.stopPropagation();
      zone.classList.remove('is-dragover');
      const files = event.dataTransfer.files;
      if (files.length !== 1) {
        message('Drop one PNG, JPG or WebP image at a time.', true);
        return;
      }
      loadFile(kind, files[0]);
    });
    $(`clear-${kind}`).addEventListener('click', (event) => {
      event.preventDefault();
      event.stopPropagation();
      clearImage(kind);
    });
  }

  for (const input of methodInputs) input.addEventListener('change', () => { render(); inputStatus(); });
  $('max-side').addEventListener('change', () => { render(); inputStatus(); });
  for (const card of cards) card.querySelector('.comparison-range').addEventListener('input', () => updateDivider(card));
  $('compare-button').addEventListener('click', (event) => {
    event.preventDefault();
    render();
    if ($('compare-button').disabled) return;
    if (state.sample) {
      inputStatus();
      return;
    }
    const count = selectedMethods().length;
    message(`Inputs are ready for ${count} ${count === 1 ? 'method' : 'methods'} at ${sizeText(processedSize())} px. Model inference is not connected in this UI preview; no restored images or measured metrics have been generated.`);
  });
  $('download-button').addEventListener('click', () => {
    if ($('download-button').disabled || !state.sample) return;
    const anchor = document.createElement('a');
    anchor.href = state.sample.download;
    anchor.download = `${state.sample.id}-saved-comparison.zip`;
    document.body.append(anchor);
    anchor.click();
    anchor.remove();
    message(`Requested the original saved comparison ZIP for ${state.sample.id}. The archive includes all three methods, regardless of the current selection.`);
  });
  $('reset-button').addEventListener('click', (event) => {
    event.preventDefault();
    clearImage('hazy', false);
    clearImage('reference', false);
    methodInputs.forEach((input) => { input.checked = METHODS.includes(input.value); });
    $('max-side').value = '512';
    for (const card of cards) card.querySelector('.comparison-range').value = '50';
    render();
    inputStatus();
  });
  for (const eventName of ['dragover', 'drop']) {
    window.addEventListener(eventName, (event) => {
      if ([...(event.dataTransfer?.types || [])].includes('Files')) event.preventDefault();
    });
  }
  window.addEventListener('pagehide', (event) => {
    if (!event.persisted) for (const url of [...activeUrls]) revoke(url);
  });
  render();
  inputStatus();
  loadManifest();
})();
