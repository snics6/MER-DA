"""gate-code の検算：dai2_grad.py の map の部分を，生徒を絞って（既定 A，V，T）CLI と同じ経路で流す（AVT は 1 単位で 666 秒かかり 10 分を超えるため）。
使い方：map_subset.py <生徒をカンマで> -- <dai2_grad.py の引数>"""
import sys

sys.path.insert(0, "/home/snics/MER-DA/DAI/exp")
import dai_data as DD  # noqa: E402

studs = tuple(sys.argv[1].split(","))
DD.STUDENTS = studs
sys.argv = ["dai2_grad.py"] + sys.argv[3:]
import dai2_grad  # noqa: E402

dai2_grad.main()
