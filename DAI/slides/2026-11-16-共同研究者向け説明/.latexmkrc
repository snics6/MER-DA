# このデッキ単体を latexmk / VS Code (LaTeX Workshop) から直接ビルドするための設定．
# リポジトリ直下の `make` を使う場合も同じ設定が効く．

# リポジトリ直下（common/ と Makefile がある場所）を上に辿って探す．
# これにより decks/<deck>/ でも decks/<project>/<deck>/ でも同じ設定で動く．
use Cwd qw(getcwd);
my $dir  = getcwd();
my $root = undef;
for (1 .. 8) {
    if (-d "$dir/common" && -f "$dir/Makefile") { $root = $dir; last; }
    last unless $dir =~ s{/[^/]+$}{};
}
die "リポジトリ直下（common/ を含む階層）が見つからない\n" unless $root;

$ENV{'TEXINPUTS'} = "$root/common//:" . ($ENV{'TEXINPUTS'} // '');
$ENV{'BIBINPUTS'} = "$root/common//:$root/../paper:" . ($ENV{'BIBINPUTS'} // '');   # 論文の references_ieee.bib を引く

$pdf_mode = 4;    # 4 = LuaLaTeX
$lualatex = 'lualatex -synctex=1 -interaction=nonstopmode -file-line-error %O %S';
$bibtex_use = 2;  # 必要なら bibtex を走らせ，clean 時に .bbl も消す

$out_dir = 'build';
$clean_ext = 'nav snm vrb synctex.gz run.xml bbl';
