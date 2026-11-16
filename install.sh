#!/bin/bash
# OT-LCDFT-RCCS のセットアップ（RCCSで最初に1回だけ実行）
#   bash install.sh
# conda環境（既定: otlcdft）を作成し、ot-g16sub を実行できるようにする。

CONDA_INIT=${OTLCDFT_CONDA_INIT:-/apl/conda/20250310/conda_init.sh}
ENV_NAME=${OTLCDFT_ENV:-otlcdft}
REPO_DIR="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"

chmod +x "$REPO_DIR/bin/ot-g16sub"

. "$CONDA_INIT" || { echo "エラー: $CONDA_INIT を読み込めません"; exit 1; }

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "conda環境 '$ENV_NAME' はすでにあります（作成をスキップ）"
else
    conda create -y -n "$ENV_NAME" python scikit-optimize matplotlib || exit 1
fi

echo
echo "セットアップが完了しました。次の1行を ~/.bashrc に追加してください（zshなら ~/.zshrc）:"
echo
echo "  export PATH=\"$REPO_DIR/bin:\$PATH\""
echo
echo "csh / tcsh の場合は ~/.cshrc に:"
echo
echo "  setenv PATH \"$REPO_DIR/bin:\${PATH}\""
echo
echo "追加したら、ログインし直す（または source ~/.bashrc）と ot-g16sub が使えます。"
