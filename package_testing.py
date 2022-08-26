import DiMo2d
#import DiMo3d
import sys


def test_2d_func():
    likelihood_dir = sys.argv[1]
    binary_dir = sys.argv[2]
    morse_dir = sys.argv[3]
    json_dir = sys.argv[4]

    ve_thresh = 0
    et_thresh = 64

    threads = 1

    DiMo2d.compute_persistence_single_channel(likelihood_dir, morse_dir, threads)
    DiMo2d.generate_morse_graphs(morse_dir, binary_dir, ve_thresh, et_thresh, threads)
    DiMo2d.postprocess_graphs(morse_dir, ve_thresh, et_thresh, threads)
    DiMo2d.cshl_post_results(morse_dir, json_dir, ve_thresh, et_thresh, threads)


if __name__ == '__main__':
    test_2d_func()
    #test_2d()
    #test_3d()
