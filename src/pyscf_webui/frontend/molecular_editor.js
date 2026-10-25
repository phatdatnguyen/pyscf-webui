// Gradio supplies element, props, watch and trigger to js_on_load. Scope every
// selector to this component so that the NGL viewer iframes remain independent.
const frame = element.querySelector(".molecule-canvas");
const useButton = element.querySelector(".use-drawing");
const status = element.querySelector(".editor-status");
let ketcher = null;
let lastLoadId = null;
let work = Promise.resolve();
let generation = 0;

function showStatus(message, error = false) {
    status.textContent = message;
    status.dataset.error = String(error);
}

function errorMessage(error) {
    return error instanceof Error ? error.message : String(error);
}

function applyLoad() {
    const command = props.value;
    if (!ketcher || !command?.load_id || command.load_id === lastLoadId) return;
    lastLoadId = command.load_id;
    const request = ++generation;
    useButton.disabled = true;
    work = work.then(async () => {
        if (request !== generation) return;
        try {
            if (command.smiles) {
                await ketcher.setMolecule(command.smiles);
            } else {
                await ketcher.editor.clear();
            }
            if (request === generation) showStatus("SMILES loaded. Edit the molecule, then choose Use drawing.");
        } catch (error) {
            if (request === generation) showStatus("Could not load this molecule into the drawing editor: " + errorMessage(error), true);
        }
    }).finally(() => {
        if (request === generation) useButton.disabled = false;
    });
    return work;
}

watch("value", applyLoad);

useButton.addEventListener("click", () => {
    if (!ketcher || useButton.disabled) return;
    const request = ++generation;
    useButton.disabled = true;
    work = work.then(async () => {
        if (request !== generation) return;
        try {
            const smiles = await ketcher.getSmiles();
            // A newer load owns the editor now; do not overwrite its SMILES field.
            if (request !== generation) return;
            props.value = {smiles, load_id: null};
            trigger("input");
            showStatus(smiles ? "Drawing sent to the SMILES field." : "Drawing cleared; SMILES field cleared.");
        } catch (error) {
            if (request === generation) showStatus("Could not read the drawing: " + errorMessage(error), true);
        }
    }).finally(() => {
        if (request === generation) useButton.disabled = false;
    });
});

if (!props.editor_available) {
    frame.hidden = true;
    showStatus("Drawing editor unavailable. You can still enter SMILES directly.", true);
} else {
    let poll;
    let timeout;
    function stopWaiting() {
        clearInterval(poll);
        clearTimeout(timeout);
        window.removeEventListener("message", onReadyMessage);
    }
    function tryReady() {
        if (!element.isConnected) {
            stopWaiting();
            return;
        }
        try {
            const editor = frame.contentWindow?.ketcher;
            if (!editor?.editor || ketcher) return;
            ketcher = editor;
            stopWaiting();
            useButton.disabled = false;
            showStatus("Draw a molecule, then choose Use drawing.");
            applyLoad();
        } catch (error) {
            stopWaiting();
            showStatus("The drawing editor could not be accessed. Reload the page or enter SMILES directly.", true);
        }
    }
    function onReadyMessage(event) {
        if (event.source === frame.contentWindow && event.origin === window.location.origin
                && event.data?.eventType === "init") {
            tryReady();
        }
    }
    window.addEventListener("message", onReadyMessage);
    poll = setInterval(tryReady, 100);
    timeout = setTimeout(() => {
        stopWaiting();
        showStatus("The drawing editor could not start. Reload the page or enter SMILES directly.", true);
    }, 30000);
    // Keep the deployment prefix, including a page URL without a trailing slash.
    const base = new URL(document.baseURI);
    if (!base.pathname.endsWith("/")) base.pathname += "/";
    base.search = "";
    base.hash = "";
    frame.src = new URL("molecule-editor/index.html?disableMacromoleculesEditor=true", base).href;
}
