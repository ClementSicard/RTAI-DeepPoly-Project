import torch
from networks import FullyConnected
from deeppoly_torch import DeepPolyVerifierTorch
from time import time
import argparse

DEVICE = "cpu"
INPUT_SIZE = 28

VERBOSE = False


def analyze(net: DeepPolyVerifierTorch, inputs: torch.Tensor, eps: float, true_label: int) -> str:
    start = time()

    """
    1. Transform initial input
    """
    initial_lower_bounds = torch.clamp(
        inputs - eps, min=0.0).to(DEVICE)
    initial_upper_bounds = torch.clamp(
        inputs + eps, max=1.0).to(DEVICE)

    """
    2. Propagate the domain through the network
    """

    with torch.no_grad():
        out, lower_bounds, upper_bounds = net(
            inputs, initial_lower_bounds, initial_upper_bounds)

    pred_label = out.max(dim=0)[1].item()
    assert pred_label == true_label

    """
    3. Verify
    """
    verified = sum((lower_bounds[true_label] > upper_bounds).int()) == 9

    end = time()
    if VERBOSE:
        print(f"Propagation done in {round(end - start, 3)}!")

    if verified:
        return verified

    """
    4. Backsubstitution if not verified by simple forward propagation
    """
    order = None
    with torch.no_grad():
        lower_bounds, upper_bounds = net.backsubstitute(
            true_label=true_label, order=order)

    verified = (lower_bounds.detach().numpy() > 0).all()
    end = time()

    if VERBOSE:
        print(f"Backsubstitution done in {round(end - start, 3)}!")

    return verified


def main():
    parser = argparse.ArgumentParser(
        description='Neural network verification using DeepPoly relaxation')
    parser.add_argument('--net',
                        type=str,
                        required=True,
                        help='Neural network architecture which is supposed to be verified.')
    parser.add_argument('--spec', type=str, required=True,
                        help='Test case to verify.')

    args = parser.parse_args()

    with open(args.spec, 'r') as f:
        lines = [line[:-1] for line in f.readlines()]
        true_label = int(lines[0])
        pixel_values = [float(line) for line in lines[1:]]
        eps = float(args.spec[:-4].split('/')[-1].split('_')[-1])

    if args.net.endswith('fc1'):
        net = FullyConnected(DEVICE, INPUT_SIZE, [50, 10]).to(DEVICE)
        deeppoly_verifier = DeepPolyVerifierTorch(
            DEVICE, INPUT_SIZE, [50, 10], verbose=VERBOSE).to(DEVICE)

    elif args.net.endswith('fc2'):
        net = FullyConnected(DEVICE, INPUT_SIZE, [100, 50, 10]).to(DEVICE)
        deeppoly_verifier = DeepPolyVerifierTorch(
            DEVICE, INPUT_SIZE, [100, 50, 10], verbose=VERBOSE).to(DEVICE)

    elif args.net.endswith('fc3'):
        net = FullyConnected(DEVICE, INPUT_SIZE, [100, 100, 10]).to(DEVICE)
        deeppoly_verifier = DeepPolyVerifierTorch(
            DEVICE, INPUT_SIZE, [100, 100, 10], verbose=VERBOSE).to(DEVICE)

    elif args.net.endswith('fc4'):
        net = FullyConnected(DEVICE, INPUT_SIZE, [100, 100, 50, 10]).to(DEVICE)
        deeppoly_verifier = DeepPolyVerifierTorch(
            DEVICE, INPUT_SIZE, [100, 100, 50, 10], verbose=VERBOSE).to(DEVICE)

    elif args.net.endswith('fc5'):
        net = FullyConnected(DEVICE, INPUT_SIZE, [
            100, 100, 100, 100, 10]).to(DEVICE)
        deeppoly_verifier = DeepPolyVerifierTorch(DEVICE, INPUT_SIZE, [
            100, 100, 100, 100, 10], verbose=VERBOSE).to(DEVICE)

    else:
        assert False

    net.load_state_dict(torch.load('../mnist_nets/%s.pt' %
                        args.net, map_location=torch.device(DEVICE)))

    inputs = torch.FloatTensor(pixel_values).view(
        1, 1, INPUT_SIZE, INPUT_SIZE).to(DEVICE)

    outs = net(inputs)
    pred_label = outs.max(dim=1)[1].item()
    assert pred_label == true_label

    deeppoly_verifier.load_weights(net=net)

    if VERBOSE:
        print(net)

    if analyze(deeppoly_verifier, inputs, eps, true_label):
        print('verified')
    else:
        print('not verified')


if __name__ == "__main__":
    try:
        main()
    except AssertionError:
        print("not verified")
