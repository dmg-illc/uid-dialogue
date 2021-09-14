"""
From https://github.com/dmg-photobook/photobook_dataset
"""

import re
import datetime
from collections import defaultdict


class Log:
    def __init__(self, logfile):
        self.game_id = logfile['game_id']
        self.domain_id = logfile['domain_id']
        self.agent_ids = logfile['agent_ids']
        self.agent_labels = logfile['agent_labels']
        self.feedback = logfile['feedback']
        self.rounds, self.complete = self.load_rounds(logfile['rounds'], self.game_id)
        self.total_score = self.calculate_score()
        self.scores = self.calculate_player_scores()
        self.start_time = logfile['start_time']
        self.duration = self.calculate_duration()
        self.domains = self.get_domains()
        self.check_feedback()

    def load_rounds(self, game_rounds, game_id):
        rounds = []
        message_id = 0
        for round_data in game_rounds:
            game_round = GameRound(round_data, game_id, message_id)
            message_id = game_round.message_id
            rounds.append(game_round)
        if len(rounds) < 5:
            return (rounds, False)
        return (rounds, True)

    def calculate_score(self):
        total_score = 0
        for game_round in self.rounds:
            total_score += game_round.total_score
        return total_score

    def calculate_player_scores(self):
        player_scores = defaultdict(lambda: 0)
        for game_round in self.rounds:
            for player, score in game_round.scores.items():
                player_scores[player] += score
        return player_scores

    def get_domains(self):
        path = self.rounds[0].images["A"][0].split("/")[0]
        return [domain for domain in path.rsplit("_", 1)]

    def calculate_duration(self):
        start_time = self.rounds[0].messages[0].timestamp
        end_time = self.rounds[-1].messages[-1].timestamp
        return end_time - start_time

    def check_feedback(self):
        if "A" not in self.feedback:
            self.feedback["A"] = None
        if "B" not in self.feedback:
            self.feedback["B"] = None

    def format_time(datetime_obj):
        return datetime_obj.strftime('%M:%S')


    def strip_image_id(image_path):
        return int(image_path.split('_')[-1].split('.')[0].lstrip('0'))




class GameRound:
    def __init__(self, logfile_entry, game_id, message_id):
        self.round_nr = logfile_entry['round_nr'] + 1
        self.images = logfile_entry['images']
        self.common = logfile_entry['common']
        self.highlighted = logfile_entry['highlighted']
        self.scores = dict(logfile_entry['score'])
        self.total_score = self.calculate_score(logfile_entry['score'])
        self.messages, self.message_id = self.load_messages(logfile_entry['messages'], message_id)
        self.num_messages = self.count_text_messages()
        self.duration = self.calculate_duration()

    def load_messages(self, message_list, message_id):
        messages = []
        for message_data in message_list:
            message = Message(message_data, message_id)
            messages.append(message)
            message_id += 1

        if len(messages) == 0:
            print('ERROR: Missing messages for this game')
        return messages, message_id

    def calculate_score(self, score_dict):
        score = 0
        if not score_dict.values():
            return None
        for player_score in score_dict.values():
            score += player_score
        return score

    def count_text_messages(self):
        count = 0
        for message in self.messages:
            if message.type == "text":
                count += 1
        return count

    def calculate_duration(self):
        start_time = self.messages[0].timestamp
        for message in self.messages[::-1]:
            if message.type == 'feedback':
                end_time = message.timestamp
                return end_time - start_time


class Message:
    def __init__(self, logfile_message, message_id):
        self.message_id = message_id
        self.agent_id = logfile_message['agent_id']
        self.text = logfile_message['message']
        self.speaker = logfile_message['speaker']
        if message_id == 0:
            self.timestamp = datetime.datetime.strptime(logfile_message['timestamp'], '%H:%M:%S')
        else:
            self.timestamp = datetime.datetime.strptime(logfile_message['timestamp'], '%H:%M:%S.%f')
        self.turn = logfile_message['turn']
        self.type = self.determine_message_type()

    def determine_message_type(self):
        if not self.text.startswith("<"):
            return "text"
        else:
            return re.findall(r'<(.*?)>', self.text)[0]
