import pickle
from typing import Any
from board import Board
from bot import Bot
from random import randint


'''
The Elo caluclator class will be a class that manages the playing of agents against each other and the counting of their elo
'''
class Elo_calculator:
    def __init__(self, game: Board, max_elo = 100, prev_elo_data =  None):
        self.game = game
        self.game_root = game
        #Dict storing key information needed to track and count a player's elo (elo, strategy), with some arb string 'name' as the key
        self.players = {}
        self.max_gain = max_elo

        #Load the data saved from the previous elo counter
        if(prev_elo_data):
            self.load(prev_elo_data)
    
    def register_bot(self, name: str, bot: Bot):
        '''
        This function will take in some strategy and name for that strategy, and create a 'player' 
        entry for it so we can track and store its elo
        '''
        if name in self.players.keys():
            print(f"Error. Name {name} is registered under this elo tracker")
            return 
        
        player = [100, bot]
        self.players[name] = player

    def save(self, file_path):
        """
        Save the current instance to a pickle file.
        """
        with open(file_path, 'wb') as file:
            pickle.dump(self.__dict__, file)
        print(f"Data saved to {file_path}")

    def load(self, file_path):
        """
        Load data from a pickle file and set it as attributes.
        """
        try:
            with open(file_path, 'rb') as file:
                data = pickle.load(file)
            if isinstance(data, dict):
                self.__dict__.update(data)
                print(f"Data loaded from {file_path}")
            else:
                raise ValueError("The pickle file does not contain a valid dictionary.")
        except Exception as e:
            print(f"Failed to load data: {e}")

    def play_match(self, name1: str, name2: str) -> str:
        """
        This function will take in the names of each player, and run a match between them, 
        returning the name of the winner, and updating each player's elo score
        """

        if name1 not in self.players.keys() or name2 not in self.players.keys():
            print("Error, not a registered player name")
            return ''
        
        #randomly select which player goes first
        if randint(1,2) % 2 == 1:
            name1, name2 = name2, name1 

        player1_elo = self.players[name1][0]
        player2_elo = self.players[name2][0]

        player1 = self.players[name1][1]
        player2 = self.players[name2][1]

        #Use the game node class to simulate the running of these 2 agents till the end of the game
        while(not self.game.is_terminal()):
            move = (0,0)
            if self.game.turn == 1:
                move = player1.choose_move(self.game)
            else:
                move = player2.choose_move(self.game)

            self.game = self.game.create_child(move)

        result = self.game.compute_winner()

        p1_new_elo, p2_new_elo = self._calc_elo(result=result, p1_elo=player1_elo, p2_elo=player2_elo)


        #Elo isn't defined under 0, so we do this
        if p1_new_elo > 0:
            self.players[name1][0] = p1_new_elo
        else:
            self.players[name1][0] = 0


        if p2_new_elo > 0:
            self.players[name2][0] = p2_new_elo
        else:
            self.players[name2][0] = 0


        #reset the game to its starting state
        self.game = self.game_root

        return name1 if result == 1 else name2
            
    def _calc_elo(self, result: int, p1_elo: float, p2_elo: float) -> tuple[float, float]:
        """
        This function will take in the result and elo of two players and calculate their updated scores
        (Assuming that result = 1 means p1 won and result = -1 means p2 won)
        """

        p1_expected = 1/(1 + 10**((p2_elo - p1_elo)/400.0))
        p2_expected = 1/(1 + 10**((p1_elo - p2_elo)/400.0))

        print(f"p1 expected {p1_expected}")
        print(f"p2 expected {p2_expected}")

        if(result == 1):
            p1_elo = p1_elo + self.max_gain*(result - p1_expected)
            p2_elo = p1_elo + self.max_gain*(-1*result - p2_expected)
        else:
            p1_elo = p1_elo + self.max_gain*(-1*result - p1_expected)
            p2_elo = p1_elo + self.max_gain*(result - p2_expected)

        print(p1_elo, p2_elo)

        return p1_elo, p2_elo
    


if __name__ == "__main__":
    
    go = Board(9)

    elo = Elo_calculator(game=go, prev_elo_data= "elo_data.pkl")
    print(elo.__dict__)

    random_player1 = Bot()
    random_player2 = Bot()

    elo.register_bot(name= "bob", strategy= random_player1)
    elo.register_bot(name = "bobette", strategy= random_player2)

    for i in range(10):
        print(elo.play_match("bob", "bobette"))

   
    elo.save("elo_data.pkl")
