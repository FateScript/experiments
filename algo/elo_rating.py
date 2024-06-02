#!/usr/bin/env python3

# Simulate a player's rating convergence in an Elo system.
# Assume the player's performance follows a normal distribution in my simulation.
# Reference: https://en.wikipedia.org/wiki/Elo_rating_system

import functools
from typing import List, Optional, Tuple
from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np


def elo_win_prob(rating1: float, rating2: float) -> float:
    """Return the win probability of player 1 againts player 2."""
    delta = rating2 - rating1
    return 1 / (1 + 10 ** (delta / 400))


def update_elo_rating(rating: float, game_result: int, win_prob: float, k: int = 32) -> float:
    """Update the player's rating based on the game result."""
    return max(rating + k * (game_result - win_prob), 0)


def game(rating1: float, rating2: float, std: float = 300) -> int:
    """Return 1 if player1 wins, 0 otherwise.
    We assume the player's performance follows a normal distribution.
    The player wins the game if his/her performance is higher than the opponent's.

    Args:
        rating1 (float): player 1 rating.
        rating2 (float): player 2 rating.
        std (float, optional): player's variance. Defaults to 300.
    """
    win = np.random.normal(rating1, std) > np.random.normal(rating2, std)
    game_result = 1 if win else 0
    return game_result


def player_converage(
    true_rating: float,
    avg_rating: float = 1200,
    std: int = 300,
    num_match_players: int = 10,
    verbose: bool = False,
) -> int:
    elo_score, steps = avg_rating, 0
    while abs(elo_score - true_rating) > 10:
        players_rating = np.random.normal(avg_rating, std, num_match_players).tolist()
        player_id = np.argmin([abs(x - elo_score) for x in players_rating])
        selected_player_rating = players_rating[player_id]

        win_prob = elo_win_prob(elo_score, selected_player_rating)
        game_result = game(true_rating, selected_player_rating)
        elo_score = update_elo_rating(elo_score, game_result, win_prob)
        if verbose:
            print(
                f"Rating: {selected_player_rating: .3f}\twin prob: {win_prob: .3f}\t",
                f"Game: {game_result}\tElo score: {elo_score: .3f}"
            )
        steps += 1
    return steps


def update_elo_rating_consider_history(history: List[int]) -> int:
    count = 1
    last_game = history[-1]
    for game in reversed(history[:-1]):
        if game == last_game:
            count += 1
        else:
            break
    value = 0
    if count > 10:
        value = 40
    elif count > 7:
        value = 20
    elif count > 5:
        value = 10
    value = value if last_game == 1 else -value
    return value


def player_converage_consider_history(
    true_rating: float,
    avg_rating: float = 1200,
    std: int = 300,
    num_match_players: int = 10,
    verbose: bool = False,
) -> int:
    elo_score, steps = avg_rating, 0
    history_games = []
    while abs(elo_score - true_rating) > 10:
        players_rating = np.random.normal(avg_rating, std, num_match_players).tolist()
        player_id = np.argmin([abs(x - elo_score) for x in players_rating])
        selected_player_rating = players_rating[player_id]

        win_prob = elo_win_prob(elo_score, selected_player_rating)
        game_result = game(true_rating, selected_player_rating)
        history_games.append(game_result)
        elo_score = update_elo_rating(elo_score, game_result, win_prob)
        elo_bonus = update_elo_rating_consider_history(history_games)
        elo_score += elo_bonus
        if verbose:
            print(
                f"Rating: {selected_player_rating: .3f}\twin prob: {win_prob: .3f}\t",
                f"Game: {game_result}\tElo score: {elo_score: .3f}\tElo bonus: {elo_bonus}"
            )
        steps += 1
    return steps


def two_players_converage(
    loops: int = 1000,
    elo_k: int = 32,
    avg_rating: float = 1200,
    player_a_rating: float = 300.0,
    a_win_b_prob: float = 0.75,
    verbose: bool = True,
) -> Tuple[List[float], float, float]:
    """
    Simulate two players' battle.

    Args:
        loops (int): Number of simulations. Defaults to 1000.
        elo_k (int): Elo K value. Defaults to 32.
        avg_rating (float): Average rating. Defaults to 1200.
        player_a_rating (float): Player A's rating. Defaults to 300.0.
        a_win_b_prob (float): The true win prob(a win b). Defaults to 0.75.

    Returns:
        Tuple[List[float], float, float]: The win prob list, A's rating, B's rating.
    """
    elo_score_a = player_a_rating
    elo_score_b = 2 * avg_rating - elo_score_a

    probs_to_plot = []
    games = []
    for _ in range(loops):
        elo_prob = elo_win_prob(elo_score_a, elo_score_b)
        probs_to_plot.append(elo_prob)
        game_result = 1 if np.random.rand() < a_win_b_prob else 0
        games.append(game_result)

        elo_score_a = update_elo_rating(elo_score_a, game_result, elo_prob, k=elo_k)
        elo_score_b = update_elo_rating(elo_score_b, 1 - game_result, 1 - elo_prob, k=elo_k)

    if verbose:
        print(f"P1: {elo_score_a: .3f}\tP2: {elo_score_b: .3f}\tWin prob: {np.mean(games): .3f}")
    return probs_to_plot, elo_score_a, elo_score_b


def two_players_converage_turns(
    elo_k: int = 32,
    avg_rating: float = 1200,
    player_a_rating: float = 300.0,
    a_win_b_prob: float = 0.75
) -> Tuple[List[float], float, float]:
    """
    Simulate two players' battle until elo prob greater than true win prob.

    Args:
        loops (int): Number of simulations. Defaults to 1000.
        elo_k (int): Elo K value. Defaults to 32.
        avg_rating (float): Average rating. Defaults to 1200.
        player_a_rating (float): Player A's rating. Defaults to 300.0.
        a_win_b_prob (float): The true win prob(a win b). Defaults to 0.75.

    Returns:
        Tuple[List[float], float, float]: The win prob list, A's rating, B's rating.
    """
    elo_score_a = player_a_rating
    elo_score_b = 2 * avg_rating - elo_score_a

    probs_to_plot = []
    games = []
    while True:
        elo_prob = elo_win_prob(elo_score_a, elo_score_b)
        probs_to_plot.append(elo_prob)
        if abs(elo_prob - a_win_b_prob) <= 0.01:
            break

        game_result = 1 if np.random.rand() < a_win_b_prob else 0
        games.append(game_result)

        elo_score_a = update_elo_rating(elo_score_a, game_result, elo_prob, k=elo_k)
        elo_score_b = update_elo_rating(elo_score_b, 1 - game_result, 1 - elo_prob, k=elo_k)

    return probs_to_plot, elo_score_a, elo_score_b


def plot(
    data: List[Tuple[List, List]],
    ratings: Optional[List] = None,
    display_text: bool = True,
    line_style: bool = False,
):
    """
    Plot the simulation results of different ratings and number of players.
    """
    colors = [
        "blue", "green", "red", "cyan",
        "magenta", "orange", "black", "purple",
        "brown", "yellow",
    ]
    half_len = len(data) // 2
    if line_style:
        used_colors = colors[:half_len]
        colors = used_colors + used_colors

    for idx, ((x, y), color) in enumerate(zip(data, colors)):
        if line_style and idx >= half_len:
            plt.plot(x, y, color=color, marker='s', linestyle="--")
        else:
            plt.plot(x, y, color=color, marker='o')

        if display_text:
            for i in range(len(x)):
                plt.text(
                    x[i], y[i], f"({x[i]}, {y[i]: .2f})",
                    color=color, ha="center", va="bottom",
                )

    plt.title("Elo rating simulation")
    plt.xlabel("Turns to converge")
    plt.ylabel("Number of players")
    if ratings:
        plt.legend(ratings, loc="upper right")

    plt.savefig("elo_rating.png")
    plt.show()


def plot_win_prob(probs: List[List[float]], true_prob: float, tags: Optional[List] = None):
    colors = [
        "blue", "green", "red", "cyan",
        "magenta", "purple", "orange",
    ]
    for color, prob in zip(colors, probs):
        x = list(range(len(prob)))
        plt.plot(x, prob, color=color)

    plt.axhline(y=true_prob, color="red", linestyle="--")

    if tags:
        plt.legend(tags, loc="upper right")

    plt.title("Win probs of two players")
    plt.xlabel("Turns")
    plt.ylabel("Probs")

    plt.savefig("prob.png")
    plt.show()


def simulate_system(num_simulation: int = 2500):
    """
    Simulate the player's rating convergence in an Elo system.
    You will the number of players will affect the convergence
    and the higher the player's rating, the slower the convergence.
    """
    mean, std = 1200, 300

    std_sacles = [0.5, 1, 1.5, 2]
    full_sacles = [-x for x in reversed(std_sacles)] + std_sacles
    ratings = [mean + scale * std for scale in full_sacles]
    num_players = [5, 10, 15, 20, 50, 80, 100, 200, 300, 400, 500]
    data = []

    for rating in tqdm(ratings):
        converge_turns = []
        for players in tqdm(num_players, leave=False):
            turns = [
                player_converage(
                    rating, num_match_players=players,
                    avg_rating=mean, std=std, verbose=False,
                )
                for _ in range(num_simulation)
            ]
            avg_turns = np.mean(turns)
            converge_turns.append(avg_turns)
        data.append((num_players, converge_turns))

    plot(data, ratings=[f"{str(x)} * sigma" for x in full_sacles])


def simulate_consider_history(num_simulation: int = 2500):
    """
    Simulate the player's rating convergence and considering the history game.
    You will see that the convergence will be faster for all levels of playes compared to
    the baseline, and the topper the player, the faster the convergence.
    """
    mean, std = 1200, 300

    std_sacles = [0.5, 1, 1.5, 2]
    ratings = [mean + scale * std for scale in std_sacles]
    num_players = [5, 10, 15, 20, 50, 80, 100, 200, 300, 400, 500]

    data, ratings_tag = [], []
    for converage_type in ["baseline", "bonus"]:
        simulate_func = player_converage if converage_type == "baseline" else player_converage_consider_history  # noqa
        for rating in tqdm(ratings):
            converge_turns = []
            for players in tqdm(num_players, leave=False):
                turns = [
                    simulate_func(
                        rating, num_match_players=players,
                        avg_rating=mean, std=std, verbose=False,
                    )
                    for _ in range(num_simulation)
                ]
                avg_turns = np.mean(turns)
                converge_turns.append(avg_turns)
            data.append((num_players, converge_turns))
        ratings_tag.extend([f"{str(x)} * sigma - {converage_type}" for x in std_sacles])

    plot(data, ratings=ratings_tag, line_style=True)


def simulate_elo_k(
    num_simulation: int = 2500,
    win_prob: float = 0.75,
    avg_rating: float = 1200.0,
    rating_a: float = 300.0,
):
    """
    Simulate two players' battle and see how the Elo K value affects the convergence.
    The two player have differnt start rating and the first player is the chanllenger.

    Args:

        num_simulation (int): Number of simulations. Defaults to 2500.
        win_prob (float): The true win prob(a win b). Defaults to 0.75.
    """
    elo_k_values = [1, 2, 4, 8, 16, 32]

    probs_list = []
    for elo_k in tqdm(elo_k_values):
        probs, *_ = two_players_converage(
            num_simulation,
            elo_k=elo_k,
            a_win_b_prob=win_prob,
            avg_rating=avg_rating,
            player_a_rating=rating_a,
        )
        probs_list.append(probs)

    # converage turns
    surpass = [[x >= win_prob for x in prob] for prob in probs_list]
    for turns, elo_k in zip(surpass, elo_k_values):
        if True not in turns:
            print(f"Elo K = {elo_k}\tNot surpass.")
            continue
        turn = turns.index(True)
        print(f"Elo K = {elo_k}\tTurns to surpass: {turn}")

    # plot image
    tags = [f"Elo K = {x}" for x in elo_k_values]
    plot_win_prob(probs_list, true_prob=win_prob, tags=tags)


def simulate_elo_k_turns(
    num_simulation: int = 1000,
    win_prob: float = 0.75,
    avg_rating: float = 1200.0,
    rating_a: float = 300.0,
):
    """
    Simulate two players' battle and see how many turns it takes under different Elo K value.
    The two player have differnt start rating and the first player is the chanllenger.
    """
    elo_k_values = [1, 2, 4, 8, 16, 32]

    turns_list, probs_list = [], []
    for elo_k in tqdm(elo_k_values):
        turns_count, last_probs = [], []
        for _ in range(num_simulation):
            probs, *_ = two_players_converage_turns(
                elo_k=elo_k,
                a_win_b_prob=win_prob,
                avg_rating=avg_rating,
                player_a_rating=rating_a,
            )
            last_probs.append(probs[-1])
            turns_count.append(len(probs))

        turns_list.append(turns_count)
        probs_list.append(last_probs)

    # converage turns
    for elo_k, turns, probs in zip(elo_k_values, turns_list, probs_list):
        avg_turns, std_turns = np.mean(turns), np.std(turns)
        avg_probs = np.mean(probs)
        print(f"Elo K = {elo_k}\tAvg Turns: {avg_turns: .2f}\t"
              f"Std: {std_turns: .2f}\tConverge prob: {avg_probs: .3%}")


def simulate_elo_k_prob_stats(
    num_simulation: int = 1000,
    num_games: int = 60,
    win_prob: float = 0.75,
    avg_rating: float = 1200.0,
    rating_a: float = 300.0,
):
    """
    Simulate two players' battle and the game end after `num_games` turns.
    See the statistics of the win prob at different turns under different Elo K value.
    """
    elo_k_values = [1, 2, 4, 8, 16, 32]
    inspect_turns = [20, 30, 40, 50, 60]

    probs_list = []
    for elo_k in tqdm(elo_k_values):
        last_probs = []
        for _ in range(num_simulation):
            probs, *_ = two_players_converage(
                num_games,
                elo_k=elo_k,
                a_win_b_prob=win_prob,
                avg_rating=avg_rating,
                player_a_rating=rating_a,
                verbose=False,
            )
            inspect_prob = [probs[turn - 1] for turn in inspect_turns]
            last_probs.append(inspect_prob)

        probs_list.append(last_probs)

    for elo_k, prob in zip(elo_k_values, probs_list):
        prob_matrix = np.array(prob).transpose()
        print(f"Elo K = {elo_k}")
        for turn, array in zip(inspect_turns, prob_matrix):
            prob_mean, prob_std = np.mean(array), np.std(array)
            print(f"Turns: {turn}\tAvg prob: {prob_mean: .2%}\tStd: {prob_std: .2%}")


if __name__ == "__main__":
    import fire
    fire.Fire({
        "system": simulate_system,
        "history": simulate_consider_history,
        "elo_k": simulate_elo_k,
        "elo_k_same_rating": functools.partial(simulate_elo_k, rating_a=1200.0),
        "elo_k_turns": simulate_elo_k_turns,  # how many turns to reach the true win prob.
        "elo_k_turns_same_rating": functools.partial(simulate_elo_k_turns, rating_a=1200.0),
        "elo_k_prob_stats": simulate_elo_k_prob_stats,  # statistics of the prob at different turns.
        "elo_k_prob_stats_same_rating": functools.partial(simulate_elo_k_prob_stats, rating_a=1200.0),  # noqa
    })
