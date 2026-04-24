from baselines.reference import run_reference_method


def main(args):
    run_reference_method(args, method="smote", mode="sample")
