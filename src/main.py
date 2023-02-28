import argparse
import logging

from algorithm import Human
from mdp import GridTargetMDP
from registry import LocalRegistry
from util.constants import *

def init_logger(level:str):
    logging.basicConfig(
        format="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y/%m/%d %H:%M:%S",
        level=logging.getLevelName(level.upper())
    )

def get_runtime_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--plan", action="store_true", help="Build the objects but do not run the agent.")
    parser.add_argument("-l", "--log-level", type=str, default="INFO", help="The logging level to use.")
    parser.add_argument("-r", "--registry", type=str, default=LOCAL_REGISTRY, help="The registry that manages I/O.")

    # registry args
    parser.add_argument("--eval-root", type=str, default="eval", help="The root directory where training artifacts are written to.")
    parser.add_argument("--data-root", type=str, default="data", help="The root directory where training data is read from.")

    mdp_parser = parser.add_subparsers(dest="mdp")

    grid_target_parser = mdp_parser.add_parser(GRID_TARGET_MDP, help="The target grid MDP.")
    grid_target_parser.add_argument("--fps", type=int, default=60, help="The frames per second.")
    grid_target_parser.add_argument("--width", type=int, default=1920, help="The display width.")
    grid_target_parser.add_argument("--height", type=int, default=1080, help="The display height.")
    grid_target_parser.add_argument("--grid-dim", type=int, default=5, help="The grid dimension.")
    grid_target_parser.add_argument("--agent-start-position", type=str, default="4,2", help="The display height.")
    grid_target_parser.add_argument("--target-start-position", type=str, default="2,4", help="The display height.")
    grid_target_parser.add_argument("--display", action="store_true", help="Display the grid on screen.")
    grid_target_parser.add_argument("--values", action="store_true", help="Display the state-action values for each state.")
    grid_target_parser.add_argument("--trail", action="store_true", help="Display a trail of the agent's path through the MDP.")

    for mdp_parser in [grid_target_parser]:
        agent_parser = mdp_parser.add_subparsers(dest="agent")
        agent_parser.add_parser(HUMAN)

    return parser.parse_args()

def main(args):
    _logger = logging.getLogger(__name__)

    _logger.info("RL Skeleton Code")
    _logger.info("=" * 50)

    _logger.info("Building registry")
    registry_name:str = args.registry
    if registry_name == LOCAL_REGISTRY:
        registry = LocalRegistry(args.eval_root, args.data_root)
    else:
        raise Exception(f"Registry {registry_name} invalid or not yet implemented.")

    _logger.info("Building MDP")
    mdp_name:str = args.mdp
    if mdp_name == GRID_TARGET_MDP:
        mdp = GridTargetMDP(args.fps,
                            args.width,
                            args.height,
                            args.grid_dim,
                            tuple(map(int, args.agent_start_position.split(","))),
                            tuple(map(int, args.target_start_position.split(","))),
                            args.display,
                            args.values,
                            args.trail)
    else:
        raise Exception(f"MDP {mdp_name} invalid or not yet implemented.")

    agent_name:str = args.agent
    if agent_name == HUMAN:
        _logger.info("Building agent")
        agent = Human(mdp)
    else:
        raise Exception(f"Agent {agent_name} invalid or not yet implemented.")

    if not args.plan:
        _logger.info("Running agent")
        agent.run()

    _logger.info("Done!")

if __name__ == "__main__":
    args = get_runtime_args()
    init_logger(args.log_level)
    main(args)
