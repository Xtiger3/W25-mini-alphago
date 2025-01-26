# Mini-AlphaGo, Winter 2025

In this project, we will build a modified hybrid of [AlphaGo](https://www.nature.com/articles/nature16961) and [AlphaZero](https://www.nature.com/articles/nature24270) for 9x9 Go from almost scratch (using only computational libraries).

## Schedule (up to today)

| **Week** | **Links** |
| --- | --- |
| 1 | [Slides](https://docs.google.com/presentation/d/1-xUB_iLC-hbhHI7JJtxdNb0yJfJjoJYizdwDDreBi8k/edit?usp=sharing), [Play Online Go](https://online-go.com/), [Computer Go Rules](https://tromp.github.io/go.html)
| 2 | TBD |

## General Usage
In board.py there's the Board class, which handles all the logic of playing the game of Go. In elo_calculator.py, there's an Elo_calculator class, which allows you to play to bots against each other, calculate their updated elos, and store them. 
In the following code:
```python
go = Board(9)
elo = Elo_calculator(game=go, prev_elo_data= "elo_data.pkl")

random_player1 = Bot()
random_player2 = Bot()

elo.register_bot(name= "bob", bot= random_player1)
elo.register_bot(name = "bobette", bot= random_player2)

elo.play_match("bob", "bobette")

elo.save("elo_data.pkl")
```
We create an elo_calculator object from past elo data, register 2 agents, and run them against each other, saving the data to 'elo_data.pkl'. If you were to run this code again, then the code below would do the same this as the code snippet above:
```python
go = Board(9)
elo = Elo_calculator(game=go, prev_elo_data= "elo_data.pkl")

elo.play_match("bob", "bobette")
elo.save("elo_data.pkl")
```
