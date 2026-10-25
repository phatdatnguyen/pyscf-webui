"""Exercise the real molecular editor and conformer workflow in Chromium.

Run explicitly; pytest does not collect this script. Requires Playwright and its
Chromium browser, an app environment, and installed Ketcher assets. The app and
Playwright may use separate Python environments; see --help. All nonlocal browser
requests are blocked throughout the test.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from urllib.request import urlopen

from playwright.sync_api import expect, sync_playwright


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", default=sys.executable,
                        help="Python executable with pyscf-webui dependencies installed")
    parser.add_argument("--package-root", type=Path,
                        default=Path(__file__).resolve().parents[2] / "src",
                        help="Package parent directory; also accepts an extracted wheel")
    parser.add_argument("--browser", help="Chromium executable; defaults to Playwright's browser")
    parser.add_argument("--browser-library-path", help="Optional Linux shared-library directory")
    parser.add_argument("--prefix", default="", help="Optional deployment prefix, e.g. /lab")
    parser.add_argument("--screenshot", type=Path, help="Optional output screenshot path")
    return parser.parse_args()


def canonical_smiles(python, values):
    """Compare chemical identity with RDKit from the app's environment."""
    result = subprocess.run([
        python, "-c",
        "import json, sys; from rdkit import Chem; "
        "print(json.dumps([Chem.MolToSmiles(Chem.MolFromSmiles(value), "
        "canonical=True, isomericSmiles=True) for value in json.load(sys.stdin)]))",
    ], input=json.dumps(values), capture_output=True, text=True, check=True)
    return json.loads(result.stdout)


def run(args):
    package_root = args.package_root.resolve()
    # Keep a virtualenv interpreter's symlink: resolving it would select system Python
    # and silently lose the app's installed dependencies.
    python = str(Path(shutil.which(args.python) or args.python).absolute())
    prefix = "/" + args.prefix.strip("/") if args.prefix.strip("/") else ""
    env = dict(os.environ, GRADIO_ANALYTICS_ENABLED="False")
    browser_env = env.copy()
    if args.browser_library_path:
        browser_env["LD_LIBRARY_PATH"] = os.pathsep.join(filter(None, (
            args.browser_library_path, browser_env.get("LD_LIBRARY_PATH"))))
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    origin = f"http://127.0.0.1:{port}"
    base = origin + prefix
    with tempfile.TemporaryDirectory(prefix="pyscf-editor-browser-") as runtime:
        run_root = Path(runtime)
        env["MPLCONFIGDIR"] = str(run_root / "matplotlib")
        with (run_root / "server.log").open("w+", encoding="utf-8") as log:
            server = subprocess.Popen([
                python, "-c",
                "import sys; sys.path.insert(0, sys.argv[1]); "
                "from pathlib import Path; import pyscf_webui; "
                "assert Path(pyscf_webui.__file__).resolve().is_relative_to(Path(sys.argv[1])); "
                "from pyscf_webui.app import build_app; "
                "from fastapi import FastAPI; inner = build_app(Path(sys.argv[2])); "
                "prefix = sys.argv[4]; "
                "app = FastAPI(lifespan=lambda _: inner.router.lifespan_context(inner)) "
                "if prefix else inner; "
                "app.mount(prefix, inner) if prefix else None; "
                "import uvicorn; uvicorn.run(app, "
                "host='127.0.0.1', port=int(sys.argv[3]), access_log=False)",
                str(package_root), runtime, str(port), prefix,
            ], cwd=runtime, env=env, stdout=log, stderr=subprocess.STDOUT)
            try:
                deadline = time.monotonic() + 60
                while time.monotonic() < deadline:
                    if server.poll() is not None:
                        raise RuntimeError("Test server exited before becoming ready")
                    try:
                        with urlopen(base + "/config", timeout=1) as response:
                            if response.status == 200:
                                break
                    except OSError:
                        time.sleep(0.25)
                else:
                    raise RuntimeError("Test server did not become ready")
                with sync_playwright() as playwright:
                    browser = playwright.chromium.launch(
                        executable_path=args.browser, headless=True, env=browser_env,
                        args=["--no-sandbox", "--disable-dev-shm-usage"],
                    )
                    context = browser.new_context(viewport={"width": 1700, "height": 1100})
                    external_requests = []

                    def local_requests_only(route):
                        if route.request.url.startswith((origin + "/", "data:", "blob:")):
                            route.continue_()
                        else:
                            external_requests.append((route.request.frame.url, route.request.url))
                            route.abort()

                    context.route("**/*", local_requests_only)
                    page = context.new_page()
                    errors = []
                    page.on("pageerror", lambda error: errors.append(str(error)))
                    page.goto(base + "/", wait_until="domcontentloaded")
                    use = page.get_by_role("button", name="Use drawing", exact=True)
                    expect(use).to_be_enabled(timeout=60000)
                    surface = page.locator("#molecular-editor")
                    status = surface.locator(".editor-status")
                    smiles = page.get_by_role("textbox", name="SMILES", exact=True)
                    load = page.get_by_role("button", name="Load SMILES into editor", exact=True)
                    frame = surface.locator("iframe").element_handle().content_frame()
                    assert frame.url.startswith(base + "/molecule-editor/"), frame.url

                    def expect_drawing(value):
                        deadline = time.monotonic() + 30
                        while frame.evaluate("async () => await window.ketcher.getSmiles()") != value:
                            if time.monotonic() >= deadline:
                                raise AssertionError(f"Editor did not load {value!r}")
                            page.wait_for_timeout(100)

                    smiles.fill("CCO")
                    load.click()
                    expect_drawing("CCO")
                    expect(status).to_contain_text("SMILES loaded")
                    print("PASS: SMILES loaded into the offline editor", flush=True)

                    frame.evaluate("window.testToken = 123")
                    smiles.fill("not-a-valid-smiles")
                    load.click()
                    expect(page.get_by_text(
                        "The SMILES string is invalid. The drawing has not been changed.",
                        exact=True,
                    )).to_be_visible()
                    expect_drawing("CCO")
                    assert frame.evaluate("window.testToken") == 123
                    print("PASS: invalid SMILES preserves the valid drawing", flush=True)

                    frame.evaluate('async () => { await window.ketcher.setMolecule("c1ccccc1O"); }')
                    use.click()
                    expect(smiles).to_have_value("Oc1ccccc1")
                    assert frame.evaluate("window.testToken") == 123
                    print("PASS: drawing exported and iframe retained", flush=True)

                    molecules = ["N[C@@H](C)C(=O)O", "F/C=C/F", "[13CH3][C@H](O)F"]
                    canonical = canonical_smiles(python, molecules)
                    for molecule, expected in zip(molecules, canonical):
                        smiles.fill(molecule)
                        load.click()
                        expect(status).to_contain_text("SMILES loaded")
                        drawn = frame.evaluate("async () => await window.ketcher.getSmiles()")
                        assert canonical_smiles(python, [drawn]) == [expected], (molecule, drawn)
                        use.click()
                        expect(status).to_contain_text("Drawing sent")
                        expect(smiles).to_have_value(expected)
                        assert frame.evaluate("window.testToken") == 123
                    print("PASS: tetrahedral and E/Z stereochemistry and isotopes preserved", flush=True)

                    smiles.fill("")
                    load.click()
                    expect_drawing("")
                    expect(use).to_be_enabled()
                    use.click()
                    expect(status).to_contain_text("Drawing cleared")
                    expect(smiles).to_have_value("")
                    print("PASS: explicit clear round trip", flush=True)

                    directory = page.get_by_role("combobox", name="Working Directory", exact=True)
                    directory.fill("browser_test")
                    directory.press("Enter")
                    page.get_by_role("button", name="Create/Open Working Directory", exact=True).click()
                    deadline = time.monotonic() + 20
                    while not (run_root / "data/browser_test").is_dir():
                        if time.monotonic() >= deadline:
                            raise AssertionError("Working directory was not created")
                        page.wait_for_timeout(100)
                    frame.evaluate('async () => { await window.ketcher.setMolecule("O"); }')
                    use.click()
                    expect(smiles).to_have_value("O")
                    page.get_by_role("button", name="Generate", exact=True).click()
                    output = run_root / "data/browser_test/conformer_1.xyz"
                    deadline = time.monotonic() + 30
                    while not output.is_file():
                        if time.monotonic() >= deadline:
                            raise AssertionError(page.locator("body").inner_text())
                        page.wait_for_timeout(100)
                    assert "O " in output.read_text(encoding="utf-8")
                    print("PASS: drawing to real conformer generation", flush=True)
                    if args.screenshot:
                        page.screenshot(path=str(args.screenshot), full_page=True)
                    assert not errors, errors
                    editor_external = [url for source, url in external_requests
                                       if "/molecule-editor/" in source]
                    assert not editor_external, editor_external
                    print(f"PASS: editor requested no external services; "
                          f"all {len(external_requests)} other external requests blocked", flush=True)
                    browser.close()
                    print(f"PASS: browser workflow with package root {package_root} "
                          f"and prefix {prefix!r}", flush=True)
            except BaseException:
                log.flush()
                log.seek(0)
                print(log.read(), file=sys.stderr)
                raise
            finally:
                server.terminate()
                try:
                    server.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    server.kill()
                    server.wait()


if __name__ == "__main__":
    run(arguments())
