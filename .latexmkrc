$clean_ext .= " pytxcode pythontex-files-%R/*";

$pythontex_cmd = 'pythontex --interpreter "python:./.venv/bin/python"';

$pdflatex = 'pdflatex -shell-escape -interaction=nonstopmode -synctex=1 %O %S && ' . $pythontex_cmd . ' %B';