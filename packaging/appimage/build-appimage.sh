#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
APP_ID="io.github.seolcu.UltimateTranscripter"
ARCH="$(uname -m)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

cd "${ROOT_DIR}"

if [[ "${ARCH}" != "x86_64" && "${ARCH}" != "aarch64" ]]; then
  echo "Unsupported architecture for AppImage build: ${ARCH}" >&2
  exit 1
fi

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "Python not found: ${PYTHON_BIN}" >&2
  exit 1
fi

if ! "${PYTHON_BIN}" -m pip --version >/dev/null 2>&1; then
  FALLBACK_PYTHON="${ROOT_DIR}/.venv/bin/python"
  if [[ -x "${FALLBACK_PYTHON}" ]] && "${FALLBACK_PYTHON}" -m pip --version >/dev/null 2>&1; then
    echo "Using fallback Python with pip: ${FALLBACK_PYTHON}" >&2
    PYTHON_BIN="${FALLBACK_PYTHON}"
  else
    echo "Selected Python does not provide pip: ${PYTHON_BIN}" >&2
    echo "Install python3-pip (dnf) or set PYTHON_BIN to a Python with pip." >&2
    exit 1
  fi
fi

if ! command -v curl >/dev/null 2>&1; then
  echo "curl is required to download appimagetool" >&2
  exit 1
fi

VERSION="$(${PYTHON_BIN} - <<'PY'
from pathlib import Path
import tomllib

pyproject = Path("pyproject.toml")
data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
print(data["project"]["version"])
PY
)"

BUILD_DIR="${ROOT_DIR}/build/appimage"
TOOLS_DIR="${BUILD_DIR}/tools"
APPDIR="${BUILD_DIR}/${APP_ID}.AppDir"
OUTPUT_PATH="${BUILD_DIR}/${APP_ID}-${VERSION}-${ARCH}.AppImage"

rm -rf "${APPDIR}"
mkdir -p "${APPDIR}/usr/bin"
mkdir -p "${APPDIR}/usr/lib/python"
mkdir -p "${APPDIR}/usr/share/applications"
mkdir -p "${APPDIR}/usr/share/icons/hicolor/scalable/apps"
mkdir -p "${APPDIR}/usr/share/metainfo"
mkdir -p "${TOOLS_DIR}"

"${PYTHON_BIN}" -m pip install --upgrade --quiet pip
"${PYTHON_BIN}" -m pip install --quiet --target "${APPDIR}/usr/lib/python" "${ROOT_DIR}"

cat > "${APPDIR}/usr/bin/ultimate-transcripter" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APPDIR="$(cd "${HERE}/../.." && pwd)"
export PYTHONPATH="${APPDIR}/usr/lib/python${PYTHONPATH:+:${PYTHONPATH}}"

exec python3 -m ultimate_transcripter.cli "$@"
EOF

cat > "${APPDIR}/usr/bin/ultimate-transcripter-gui" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APPDIR="$(cd "${HERE}/../.." && pwd)"
export PYTHONPATH="${APPDIR}/usr/lib/python${PYTHONPATH:+:${PYTHONPATH}}"

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "Error: ffmpeg is required on the host system." >&2
  exit 1
fi
if ! command -v ffprobe >/dev/null 2>&1; then
  echo "Error: ffprobe is required on the host system." >&2
  exit 1
fi

python3 - <<'PY' || {
import gi
gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")
from gi.repository import Gtk, Adw  # noqa: F401
PY
  echo "Error: GTK4/libadwaita Python bindings are required on the host system." >&2
  echo "Install packages such as python3-gobject, gtk4, and libadwaita." >&2
  exit 1
}

exec python3 -m ultimate_transcripter.gui "$@"
EOF

cat > "${APPDIR}/AppRun" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${HERE}/usr/bin/ultimate-transcripter-gui" "$@"
EOF

chmod +x "${APPDIR}/AppRun"
chmod +x "${APPDIR}/usr/bin/ultimate-transcripter"
chmod +x "${APPDIR}/usr/bin/ultimate-transcripter-gui"

DESKTOP_SRC="${ROOT_DIR}/packaging/flatpak/${APP_ID}.desktop"
ICON_SRC="${ROOT_DIR}/packaging/flatpak/${APP_ID}.svg"
METAINFO_SRC="${ROOT_DIR}/packaging/flatpak/${APP_ID}.metainfo.xml"

cp "${DESKTOP_SRC}" "${APPDIR}/${APP_ID}.desktop"
cp "${DESKTOP_SRC}" "${APPDIR}/usr/share/applications/${APP_ID}.desktop"
cp "${ICON_SRC}" "${APPDIR}/${APP_ID}.svg"
cp "${ICON_SRC}" "${APPDIR}/usr/share/icons/hicolor/scalable/apps/${APP_ID}.svg"
cp "${METAINFO_SRC}" "${APPDIR}/usr/share/metainfo/${APP_ID}.metainfo.xml"
cp "${METAINFO_SRC}" "${APPDIR}/usr/share/metainfo/${APP_ID}.appdata.xml"

if command -v appimagetool >/dev/null 2>&1; then
  APPIMAGETOOL_BIN="$(command -v appimagetool)"
else
  APPIMAGETOOL_BIN="${TOOLS_DIR}/appimagetool-${ARCH}.AppImage"
  if [[ ! -x "${APPIMAGETOOL_BIN}" ]]; then
    if [[ "${ARCH}" == "x86_64" ]]; then
      APPIMAGETOOL_URL="https://github.com/AppImage/AppImageKit/releases/download/continuous/appimagetool-x86_64.AppImage"
    else
      APPIMAGETOOL_URL="https://github.com/AppImage/AppImageKit/releases/download/continuous/appimagetool-aarch64.AppImage"
    fi

    curl -L "${APPIMAGETOOL_URL}" -o "${APPIMAGETOOL_BIN}"
    chmod +x "${APPIMAGETOOL_BIN}"
  fi
fi

APPIMAGE_EXTRACT_AND_RUN=1 ARCH="${ARCH}" "${APPIMAGETOOL_BIN}" "${APPDIR}" "${OUTPUT_PATH}"

echo "AppImage created: ${OUTPUT_PATH}"
