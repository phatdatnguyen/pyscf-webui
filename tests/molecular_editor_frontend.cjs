// Run the shipped js_on_load against the documented Gradio interface and a
// controlled asynchronous Ketcher, so races do not depend on browser timing.
const assert = require("node:assert/strict");
const fs = require("node:fs");
const test = require("node:test");
const vm = require("node:vm");
const source = fs.readFileSync(process.argv[2], "utf8");

function harness({available = true, baseURI = "http://localhost/", value} = {}) {
    const frame = {contentWindow: {}, hidden: false};
    const buttonEvents = {};
    const button = {disabled: true, addEventListener: (name, callback) => buttonEvents[name] = callback};
    const status = {textContent: "", dataset: {}};
    const elements = {".molecule-canvas": frame, ".use-drawing": button, ".editor-status": status};
    const element = {isConnected: true, querySelector: selector => elements[selector]};
    const props = {editor_available: available, value: value ?? {smiles: "", load_id: null}};
    const watchers = {};
    const events = [];
    const listeners = {};
    const intervals = new Map();
    const timeouts = new Map();
    let nextTimer = 0;
    const window = {
        location: {origin: "http://localhost"},
        addEventListener: (name, callback) => listeners[name] = callback,
        removeEventListener: name => delete listeners[name],
    };
    vm.runInNewContext(source, {
        element, props, window, URL, document: {baseURI},
        watch: (name, callback) => watchers[name] = callback,
        trigger: name => events.push(name),
        setInterval: callback => {const id = ++nextTimer; intervals.set(id, callback); return id;},
        clearInterval: id => intervals.delete(id),
        setTimeout: callback => {const id = ++nextTimer; timeouts.set(id, callback); return id;},
        clearTimeout: id => timeouts.delete(id),
    });
    return {
        frame, button, status, element, props, events, listeners, intervals, timeouts,
        ready(ketcher) {
            frame.contentWindow.ketcher = ketcher;
            for (const callback of [...intervals.values()]) callback();
        },
        load(smiles, load_id) {
            props.value = {smiles, load_id};
            return watchers.value();
        },
        click: () => buttonEvents.click(),
    };
}

function deferred() {
    let resolve;
    const promise = new Promise(done => resolve = done);
    return {promise, resolve};
}

async function settle() {
    for (let i = 0; i < 12; i++) await Promise.resolve();
}

test("absent assets leave direct SMILES usable without loading an iframe", () => {
    const h = harness({available: false});
    assert.equal(h.frame.hidden, true);
    assert.equal(h.frame.src, undefined);
    assert.equal(h.button.disabled, true);
    assert.match(h.status.textContent, /enter SMILES directly/);
    assert.equal(h.intervals.size, 0);
});

test("asset URL preserves deployment prefixes without a trailing slash", () => {
    const h = harness({baseURI: "http://localhost/chem?theme=dark#tab"});
    assert.equal(h.frame.src, "http://localhost/chem/molecule-editor/index.html?disableMacromoleculesEditor=true");
});

test("loads queued before readiness and repeated explicit loads are applied", async () => {
    const h = harness({value: {smiles: "CCO", load_id: "first"}});
    const loaded = [];
    h.ready({editor: {clear: () => loaded.push("")}, setMolecule: value => loaded.push(value)});
    await settle();
    await h.load("CCO", "second");
    await h.load("CCO", "second");
    await h.load("", "clear");
    assert.deepEqual(loaded, ["CCO", "CCO", ""]);
    assert.equal(h.button.disabled, false);
    assert.equal(h.intervals.size, 0);
    assert.equal(h.timeouts.size, 0);
});

test("export uses Gradio input and empty drawings clear SMILES", async () => {
    const h = harness();
    let molecule = "CCO";
    h.ready({editor: {}, getSmiles: () => molecule});
    h.click();
    await settle();
    assert.equal(h.props.value.smiles, "CCO");
    assert.equal(h.props.value.load_id, null);
    assert.deepEqual(h.events, ["input"]);
    molecule = "";
    h.click();
    await settle();
    assert.equal(h.props.value.smiles, "");
    assert.match(h.status.textContent, /field cleared/);
});

test("a late export cannot overwrite a newer load", async () => {
    const h = harness();
    const reading = deferred();
    const loaded = [];
    h.ready({editor: {}, getSmiles: () => reading.promise, setMolecule: value => loaded.push(value)});
    h.click();
    await settle();
    const loading = h.load("CCN", "newer-load");
    reading.resolve("CCO");
    await loading;
    assert.deepEqual(h.events, []);
    assert.deepEqual(loaded, ["CCN"]);
    assert.equal(h.props.value.smiles, "CCN");
    assert.equal(h.button.disabled, false);
});

test("failed loads recover and allow an explicit retry", async () => {
    const h = harness();
    let fail = true;
    h.ready({editor: {}, setMolecule: () => {if (fail) throw null;}});
    await h.load("CCO", "first");
    assert.equal(h.status.dataset.error, "true");
    assert.equal(h.button.disabled, false);
    fail = false;
    await h.load("CCO", "retry");
    assert.equal(h.status.dataset.error, "false");
    assert.match(h.status.textContent, /SMILES loaded/);
});

test("inaccessible iframe stops polling with an actionable error", () => {
    const h = harness();
    Object.defineProperty(h.frame.contentWindow, "ketcher", {get() {throw new Error("cross origin");}});
    for (const callback of [...h.intervals.values()]) callback();
    assert.equal(h.status.dataset.error, "true");
    assert.match(h.status.textContent, /could not be accessed/);
    assert.equal(h.intervals.size, 0);
    assert.equal(h.timeouts.size, 0);
});

test("readiness messages must match iframe source and page origin", () => {
    const h = harness();
    h.frame.contentWindow.ketcher = {editor: {}};
    h.listeners.message({source: {}, origin: "http://localhost", data: {eventType: "init"}});
    assert.equal(h.button.disabled, true);
    h.listeners.message({source: h.frame.contentWindow, origin: "https://other", data: {eventType: "init"}});
    assert.equal(h.button.disabled, true);
    h.listeners.message({source: h.frame.contentWindow, origin: "http://localhost", data: {eventType: "init"}});
    assert.equal(h.button.disabled, false);
});

test("removed component stops its startup timers", () => {
    const h = harness();
    h.element.isConnected = false;
    for (const callback of [...h.intervals.values()]) callback();
    assert.equal(h.intervals.size, 0);
    assert.equal(h.timeouts.size, 0);
});
