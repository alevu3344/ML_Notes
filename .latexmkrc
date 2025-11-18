$clean_ext .= " pytxcode pythontex-files-%R/*";

if (-e "./.venv/bin/python") {
    $pythontex_cmd = 'pythontex --interpreter "python:./.venv/bin/python"';
} else {
    $pythontex_cmd = 'pythontex';
}

$pdflatex = 'pdflatex -shell-escape -interaction=nonstopmode -synctex=1 %O %S && ' . $pythontex_cmd . ' %B';