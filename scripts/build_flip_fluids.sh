#!/usr/bin/env bash
# Build FLIP Fluids v1.8.9 from source (clean unwatermarked GitHub release)
# Produces both a legacy folder zip (flip_fluids_addon_1.8.9_linux_x64.zip) and
# a Blender 4.2+/5.x Extension CLI-ready zip (flip_fluids_addon_1.8.9_linux_x64_ext.zip).

set -euo pipefail

TAG="v1.8.9"
REPO_URL="https://github.com/rlguy/Blender-FLIP-Fluids.git"
WITH_ALEMBIC="OFF"
NATIVE_PATCH="ON"
MARCH_NATIVE="OFF"
STRIP_WATERMARK="ON"
DISTRIBUTE_SOURCE="OFF"
CLEAN_BUILD="OFF"
SRC_DIR="/tmp/Blender-FLIP-Fluids"
OUT_DIR="$(pwd)/dist"
BLENDER_BIN=""

default_jobs() {
    local cpus
    cpus="$(nproc 2>/dev/null || echo 4)"
    if [ "$cpus" -gt 12 ]; then
        echo 12
    else
        echo "$cpus"
    fi
}

JOBS="$(default_jobs)"

usage() {
    cat <<EOF
Usage: $(basename "$0") [options]

Build rlguy/Blender-FLIP-Fluids from source for Linux x86_64.

Options:
  --with-alembic         Enable Alembic support (-DWITH_ALEMBIC=ON; installs Alembic/Imath if missing)
  --without-alembic      Disable Alembic support (-DWITH_ALEMBIC=OFF) [default]
  --native               Patch out force_link_glibc_2.5.h for modern GCC/GLIBC [default]
  --no-native            Keep upstream force_link_glibc_2.5.h include
  --march-native         Pass -march=native to C++ compiler flags
  --keep-watermark       Keep upstream 3D Demo watermark mesh in fluidsimulation.cpp
  --jobs N               Parallel build jobs (default: min(\$(nproc), 12) = ${JOBS})
  --out-dir DIR          Directory for output zip packages (default: ${OUT_DIR})
  --src-dir DIR          Directory to clone FLIP Fluids source (default: ${SRC_DIR})
  --tag TAG              Git tag to build (default: ${TAG})
  --distribute-source    Include source archive inside addon zip (-DDISTRIBUTE_SOURCE=ON)
  --clean                Remove existing CMake build directory before configuring
  --blender-bin PATH     Optional path to blender binary for 'extension validate' check
  -h, --help             Show this help message
EOF
}

while [ $# -gt 0 ]; do
    case "$1" in
        --with-alembic)
            WITH_ALEMBIC="ON"
            shift
            ;;
        --without-alembic)
            WITH_ALEMBIC="OFF"
            shift
            ;;
        --native)
            NATIVE_PATCH="ON"
            shift
            ;;
        --no-native)
            NATIVE_PATCH="OFF"
            shift
            ;;
        --march-native)
            MARCH_NATIVE="ON"
            shift
            ;;
        --keep-watermark)
            STRIP_WATERMARK="OFF"
            shift
            ;;
        --jobs)
            [ $# -ge 2 ] || { echo "ERROR: --jobs requires an argument" >&2; exit 2; }
            JOBS="$2"
            shift 2
            ;;
        --out-dir)
            [ $# -ge 2 ] || { echo "ERROR: --out-dir requires an argument" >&2; exit 2; }
            OUT_DIR="$2"
            shift 2
            ;;
        --src-dir)
            [ $# -ge 2 ] || { echo "ERROR: --src-dir requires an argument" >&2; exit 2; }
            SRC_DIR="$2"
            shift 2
            ;;
        --tag)
            [ $# -ge 2 ] || { echo "ERROR: --tag requires an argument" >&2; exit 2; }
            TAG="$2"
            shift 2
            ;;
        --distribute-source)
            DISTRIBUTE_SOURCE="ON"
            shift
            ;;
        --clean)
            CLEAN_BUILD="ON"
            shift
            ;;
        --blender-bin)
            [ $# -ge 2 ] || { echo "ERROR: --blender-bin requires an argument" >&2; exit 2; }
            BLENDER_BIN="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "ERROR: Unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

run_apt_install() {
    local pkgs=("$@")
    if [ "${#pkgs[@]}" -eq 0 ]; then
        return 0
    fi
    if [ "$(id -u)" -eq 0 ]; then
        DEBIAN_FRONTEND=noninteractive apt-get update -qq
        DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends "${pkgs[@]}"
    elif command -v sudo >/dev/null 2>&1 && sudo -n true 2>/dev/null; then
        sudo DEBIAN_FRONTEND=noninteractive apt-get update -qq
        sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends "${pkgs[@]}"
    else
        echo "ERROR: Missing system packages (${pkgs[*]}) and cannot run apt-get without root/passwordless sudo." >&2
        exit 1
    fi
}

echo "[build_flip_fluids] Configuration:"
echo "  Tag               : ${TAG}"
echo "  Source dir        : ${SRC_DIR}"
echo "  Output dir        : ${OUT_DIR}"
echo "  Jobs              : ${JOBS}"
echo "  WITH_ALEMBIC      : ${WITH_ALEMBIC}"
echo "  Native glibc patch: ${NATIVE_PATCH}"
echo "  -march=native     : ${MARCH_NATIVE}"
echo "  DISTRIBUTE_SOURCE : ${DISTRIBUTE_SOURCE}"

for tool in git cmake g++ make python3; do
    if ! command -v "$tool" >/dev/null 2>&1; then
        echo "ERROR: Required build tool not found on PATH: $tool" >&2
        exit 1
    fi
done

if [ "$WITH_ALEMBIC" = "ON" ]; then
    if ! ldconfig -p 2>/dev/null | grep -q "libAlembic" && [ ! -d "/usr/include/Alembic" ]; then
        echo "[build_flip_fluids] Alembic requested but not found; installing libalembic-dev libimath-dev..."
        run_apt_install libalembic-dev libimath-dev
    fi
fi

mkdir -p "$(dirname "$SRC_DIR")" "$OUT_DIR"

if [ -d "${SRC_DIR}/.git" ]; then
    echo "[build_flip_fluids] Reusing existing git repository at ${SRC_DIR}..."
    git -C "$SRC_DIR" checkout -- CMakeLists.txt src/addon/blender_manifest.toml.in src/addon/__init__.py.in src/engine/fluidsimulation.cpp || true
    CURRENT_TAG="$(git -C "$SRC_DIR" describe --tags --exact-match 2>/dev/null || true)"
    if [ "$CURRENT_TAG" != "$TAG" ]; then
        echo "[build_flip_fluids] Switching repository to tag ${TAG}..."
        git -C "$SRC_DIR" fetch --depth 1 origin "refs/tags/${TAG}:refs/tags/${TAG}"
        git -C "$SRC_DIR" checkout -f "$TAG"
    fi
elif [ -e "$SRC_DIR" ] && [ -n "$(ls -A "$SRC_DIR" 2>/dev/null || true)" ]; then
    echo "ERROR: --src-dir (${SRC_DIR}) exists and is non-empty without .git" >&2
    exit 1
else
    echo "[build_flip_fluids] Cloning ${REPO_URL} (${TAG})..."
    git clone --depth 1 --branch "$TAG" "$REPO_URL" "$SRC_DIR"
fi

CMAKE_FILE="${SRC_DIR}/CMakeLists.txt"
MANIFEST_IN="${SRC_DIR}/src/addon/blender_manifest.toml.in"
INIT_IN="${SRC_DIR}/src/addon/__init__.py.in"
FLUIDSIM_CPP="${SRC_DIR}/src/engine/fluidsimulation.cpp"
if [ ! -f "$CMAKE_FILE" ]; then
    echo "ERROR: Missing CMakeLists.txt at ${CMAKE_FILE}" >&2
    exit 1
fi

echo "[build_flip_fluids] Patching source metadata (<=64 char manifest tagline, Source license, unwatermarked engine)..."
python3 - "$CMAKE_FILE" "$MANIFEST_IN" "$INIT_IN" "$FLUIDSIM_CPP" "$NATIVE_PATCH" "$STRIP_WATERMARK" <<'PYEOF'
import pathlib, re, sys

cmake_path = pathlib.Path(sys.argv[1])
manifest_in = pathlib.Path(sys.argv[2])
init_in = pathlib.Path(sys.argv[3])
fluidsim_cpp = pathlib.Path(sys.argv[4])
native_patch = sys.argv[5] == "ON"
strip_watermark = sys.argv[6] == "ON"

cmake_text = cmake_path.read_text(encoding="utf-8")
if native_patch:
    pattern = re.compile(
        r'^\s*set\s*\(\s*CMAKE_CXX_FLAGS\s+[^\n]*force_link_glibc_[^\n]*\)\s*$',
        re.MULTILINE,
    )
    cmake_text, count = pattern.subn(
        "# [patched by Blender-Colab --native] removed force_link_glibc_2.5.h",
        cmake_text,
    )
    print(f"[build_flip_fluids] Patched {count} force_link_glibc line(s) in CMakeLists.txt.")

cmake_text, lic_count = re.subn(
    r'set\(\s*FFENGINE_VERSION_SUPPORT_LICENSE_TYPE\s+"Demo"\s*\)',
    'set(FFENGINE_VERSION_SUPPORT_LICENSE_TYPE "Source")',
    cmake_text,
)
cmake_path.write_text(cmake_text, encoding="utf-8")
if lic_count:
    print("[build_flip_fluids] Updated FFENGINE_VERSION_SUPPORT_LICENSE_TYPE -> Source.")

if manifest_in.is_file():
    m_text = manifest_in.read_text(encoding="utf-8")
    short_tagline = (
        'tagline = "FLIP Fluid Simulation Tool for Blender '
        '(v@FFENGINE_VERSION_MAJOR@.@FFENGINE_VERSION_MINOR@.@FFENGINE_VERSION_REVISION@)"'
    )
    m_text, tag_count = re.subn(r'^tagline\s*=.*$', short_tagline, m_text, flags=re.MULTILINE)
    manifest_in.write_text(m_text, encoding="utf-8")
    print(f"[build_flip_fluids] Patched {tag_count} tagline line(s) in blender_manifest.toml.in (<=64 chars).")

if init_in.is_file():
    i_text = init_in.read_text(encoding="utf-8")
    i_text = i_text.replace('"name" : "FLIP Fluids Demo"', '"name" : "FLIP Fluids"')
    init_in.write_text(i_text, encoding="utf-8")

if strip_watermark and fluidsim_cpp.is_file():
    cpp_text = fluidsim_cpp.read_text(encoding="utf-8")
    pat_arrays = re.compile(
        r"(#if defined\(_MSC_VER\)\s*\n\s*#pragma warning\(push\).*?_triz = _trix;)",
        re.DOTALL,
    )
    pat_append = re.compile(
        r"([ \t]*AABB bbox = _getBoundaryAABB\(\);\s*\n\s*bbox\.expand\(-8\.0f \* _dx\);.*?previewmesh = previewAddMesh;\s*\n[ \t]*\})",
        re.DOTALL,
    )
    cpp_text, c_arr = pat_arrays.subn(
        "    // [patched by Blender-Colab] stripped embedded demo watermark mesh arrays (_vertx/_trix)",
        cpp_text,
    )
    cpp_text, c_app = pat_append.subn(
        "    // [patched by Blender-Colab] stripped demo watermark addmesh injection into surface/preview",
        cpp_text,
    )
    assert "surfacemesh.append(addmesh)" not in cpp_text, "Failed to strip surfacemesh.append(addmesh)!"
    assert "previewmesh.append(addmesh)" not in cpp_text, "Failed to strip previewmesh.append(addmesh)!"
    fluidsim_cpp.write_text(cpp_text, encoding="utf-8")
    print(f"[build_flip_fluids] Stripped demo watermark mesh from fluidsimulation.cpp (arrays={c_arr}, append={c_app}).")
PYEOF

BUILD_DIR="${SRC_DIR}/build"
if [ "$CLEAN_BUILD" = "ON" ] && [ -d "$BUILD_DIR" ]; then
    echo "[build_flip_fluids] Cleaning existing build directory ${BUILD_DIR}..."
    rm -rf "$BUILD_DIR"
fi
mkdir -p "$BUILD_DIR"

CMAKE_ARGS=(
    -S "$SRC_DIR"
    -B "$BUILD_DIR"
    -DCMAKE_BUILD_TYPE=Release
    "-DWITH_ALEMBIC=${WITH_ALEMBIC}"
    "-DDISTRIBUTE_SOURCE=${DISTRIBUTE_SOURCE}"
    -DDISTRIBUTE_MEDIA=OFF
)
if [ "$MARCH_NATIVE" = "ON" ]; then
    CMAKE_ARGS+=("-DCMAKE_CXX_FLAGS=-march=native")
fi

echo "[build_flip_fluids] Running CMake configure..."
cmake "${CMAKE_ARGS[@]}"

echo "[build_flip_fluids] Building libffengine.so with ${JOBS} parallel jobs..."
cmake --build "$BUILD_DIR" --parallel "$JOBS"

ADDON_DIR="${BUILD_DIR}/bl_flip_fluids/flip_fluids_addon"
ENGINE_SO="${ADDON_DIR}/ffengine/lib/libffengine.so"
MANIFEST_TOML="${ADDON_DIR}/blender_manifest.toml"

if [ ! -s "$ENGINE_SO" ]; then
    echo "ERROR: Compiled engine library missing or empty: ${ENGINE_SO}" >&2
    exit 1
fi
if [ ! -f "$MANIFEST_TOML" ]; then
    echo "ERROR: Configured blender_manifest.toml missing: ${MANIFEST_TOML}" >&2
    exit 1
fi

echo "[build_flip_fluids] Compiled engine library verified: ${ENGINE_SO} ($(stat -c%s "$ENGINE_SO") bytes)"
if command -v ldd >/dev/null 2>&1; then
    MISSING_SO="$(ldd "$ENGINE_SO" 2>/dev/null | grep "not found" || true)"
    if [ -n "$MISSING_SO" ]; then
        echo "ERROR: Unresolved shared library dependencies in ${ENGINE_SO}:" >&2
        echo "$MISSING_SO" >&2
        exit 1
    fi
fi

if [ "$STRIP_WATERMARK" = "ON" ]; then
    mkdir -p "${ADDON_DIR}/resources/installation_data"
    printf '1\n' > "${ADDON_DIR}/resources/installation_data/unwatermarked_build"
else
    rm -f "${ADDON_DIR}/resources/installation_data/unwatermarked_build"
fi

VER_CLEAN="${TAG#v}"
OUT_DIR_ABS="$(cd "$OUT_DIR" && pwd)"
FOLDER_ZIP="${OUT_DIR_ABS}/flip_fluids_addon_${VER_CLEAN}_linux_x64.zip"
EXT_ZIP="${OUT_DIR_ABS}/flip_fluids_addon_${VER_CLEAN}_linux_x64_ext.zip"
STAGING_FOLDER_ZIP="${BUILD_DIR}/flip_fluids_addon_${VER_CLEAN}_linux_x64.zip.staging"
STAGING_EXT_ZIP="${BUILD_DIR}/flip_fluids_addon_${VER_CLEAN}_linux_x64_ext.zip.staging"
rm -f "$STAGING_FOLDER_ZIP" "$STAGING_EXT_ZIP"

echo "[build_flip_fluids] Packaging folder zip and Blender Extension zip..."
python3 - "$ADDON_DIR" "$STAGING_FOLDER_ZIP" "$STAGING_EXT_ZIP" <<'PYEOF'
import os
import pathlib
import stat
import sys
import zipfile

addon_dir = pathlib.Path(sys.argv[1]).resolve()
folder_zip = pathlib.Path(sys.argv[2]).resolve()
ext_zip = pathlib.Path(sys.argv[3]).resolve()


def write_zip(src_root: pathlib.Path, out_zip: pathlib.Path, include_top_folder: bool) -> None:
    if out_zip.exists():
        out_zip.unlink()
    base_root = src_root.parent if include_top_folder else src_root
    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for root, dirs, files in os.walk(src_root):
            dirs.sort()
            files.sort()
            root_path = pathlib.Path(root)
            if "__pycache__" in root_path.parts:
                continue
            for fname in files:
                if fname.endswith((".pyc", ".pyo")):
                    continue
                fpath = root_path / fname
                arcname = fpath.relative_to(base_root).as_posix()
                st = fpath.stat()
                zinfo = zipfile.ZipInfo.from_file(fpath, arcname)
                zinfo.external_attr = (stat.S_IMODE(st.st_mode) & 0xFFFF) << 16
                zinfo.compress_type = zipfile.ZIP_DEFLATED
                with fpath.open("rb") as fh:
                    zf.writestr(zinfo, fh.read())


write_zip(addon_dir, folder_zip, include_top_folder=True)
write_zip(addon_dir, ext_zip, include_top_folder=False)

with zipfile.ZipFile(folder_zip, "r") as zf:
    names = set(zf.namelist())
    assert "flip_fluids_addon/blender_manifest.toml" in names, "Missing flip_fluids_addon/blender_manifest.toml in folder zip"
    assert "flip_fluids_addon/ffengine/lib/libffengine.so" in names, "Missing libffengine.so in folder zip"

with zipfile.ZipFile(ext_zip, "r") as zf:
    names = set(zf.namelist())
    assert "blender_manifest.toml" in names, "Missing root blender_manifest.toml in extension zip"
    assert "ffengine/lib/libffengine.so" in names, "Missing ffengine/lib/libffengine.so in extension zip"
    manifest_text = zf.read("blender_manifest.toml").decode("utf-8")
    for line in manifest_text.splitlines():
        if line.strip().startswith("tagline"):
            val = line.split("=", 1)[1].strip().strip('"').strip("'")
            assert len(val) <= 64, f"blender_manifest.toml tagline exceeds 64 chars ({len(val)}): {val!r}"
PYEOF

if [ -z "$BLENDER_BIN" ] && command -v blender >/dev/null 2>&1; then
    BLENDER_BIN="$(command -v blender)"
fi

if [ -n "$BLENDER_BIN" ] && [ -x "$BLENDER_BIN" ]; then
    echo "[build_flip_fluids] Validating extension package with ${BLENDER_BIN}..."
    if ! "$BLENDER_BIN" --command extension validate "$STAGING_EXT_ZIP"; then
        rm -f "$STAGING_FOLDER_ZIP" "$STAGING_EXT_ZIP"
        echo "ERROR: Blender extension validation failed; staging archives removed." >&2
        exit 1
    fi
fi

mv -f "$STAGING_FOLDER_ZIP" "$FOLDER_ZIP"
mv -f "$STAGING_EXT_ZIP" "$EXT_ZIP"
echo "[build_flip_fluids] Folder zip   : ${FOLDER_ZIP} ($(stat -c%s "$FOLDER_ZIP") bytes)"
echo "[build_flip_fluids] Extension zip: ${EXT_ZIP} ($(stat -c%s "$EXT_ZIP") bytes)"

echo "FLIP_ZIP_FOLDER=${FOLDER_ZIP}"
echo "FLIP_ZIP_EXT=${EXT_ZIP}"
echo "[build_flip_fluids] Done."
