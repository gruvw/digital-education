import json
import csv
from typing import Callable, Dict, Iterable, List, TypeVar
from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import defaultdict


SURVEY_CSV = "data.csv"
GRADING_JSON = "grading.json"


T = TypeVar("T")
U = TypeVar("U")
K = TypeVar("K")


@dataclass
class Answer:
    index: int

    start_time: datetime
    end_time: datetime
    duration: timedelta

    gender: str # Male, Female, Prefer not to say, Other (specified)
    age_group: str # ≤18, 19-21, 22-24, 25-27, ≥28

    pre_dfa_words: List[str]
    pre_dfa_words_grade: int # from 0 to 8 (negative points)
    pre_dfa_words_score: float # from 0 to 1
    pre_dfa_property: str
    pre_dfa_property_grade: int # 0: incorrect, 1: partially correct, 2: exact
    pre_dfa_property_score: float # from 0 to 1
    pre_blanks: List[str]
    pre_blanks_grade: int # from 0 to 3
    pre_blanks_score: float # from 0 to 1
    pre_score: float # between 0 & 1

    activity_type: str # I-PS, PS-I
    ps_link: str

    post_dfa_words: List[str]
    post_dfa_words_grade: int # from 0 to 8 (negative points)
    post_dfa_words_score: float # from 0 to 1
    post_dfa_property: str
    post_dfa_property_grade: int # 0: incorrect, 1: partially correct, 2: exact
    post_dfa_property_score: float # from 0 to 1
    post_blanks: List[str]
    post_blanks_grade: int # from 0 to 3
    post_blanks_score: float # from 0 to 1
    post_score: float # between 0 & 1


    @property
    def learning_gain(self) -> float:
        return self.post_score - self.pre_score


    @property
    def relative_learning_gain(self) -> float:
        if self.pre_score >= 1.0:
            return 0.0
        return (self.post_score - self.pre_score) / (1.0 - self.pre_score)


    def __str__(self):
        SEP = " / "

        fields = [
            f"Index: {self.index}",
            f"Start Time: {self.start_time}",
            f"End Time: {self.end_time}",
            f"Duration: {self.duration}",
            f"Gender: {self.gender}",
            f"Age Group: {self.age_group}",
            f"Pre-Test DFA Words: {SEP.join(self.pre_dfa_words)}",
            f"Pre-Test DFA Words Grade: {self.pre_dfa_words_grade}",
            f"Pre-Test DFA Words Score: {self.pre_dfa_words_score}",
            f"Pre-Test DFA Property: {truncate(self.pre_dfa_property, 60)}",
            f"Pre-Test DFA Property Grade: {self.pre_dfa_property_grade}",
            f"Pre-Test DFA Property Score: {self.pre_dfa_property_score}",
            f"Pre-Test Blanks: {SEP.join(self.pre_blanks)}",
            f"Pre-Test Blanks Grade: {self.pre_blanks_grade}",
            f"Pre-Test Blanks Score: {self.pre_blanks_score}",
            f"Pre-Test Score: {self.pre_score}",
            f"Activity Type: {self.activity_type}",
            # f"Full PS Link: {self.ps_link}",
            f"PS Link: {truncate(self.ps_link, 40)}",
            f"Post-Test DFA Words: {SEP.join(self.post_dfa_words)}",
            f"Post-Test DFA Words Grade: {self.post_dfa_words_grade}",
            f"Post-Test DFA Words Score: {self.post_dfa_words_score}",
            f"Post-Test DFA Property: {truncate(self.post_dfa_property, 60)}",
            f"Post-Test DFA Property Grade: {self.post_dfa_property_grade}",
            f"Post-Test DFA Property Score: {self.post_dfa_property_score}",
            f"Post-Test Blanks: {SEP.join(self.post_blanks)}",
            f"Post-Test Blanks Grade: {self.post_blanks_grade}",
            f"Post-Test Blanks Score: {self.post_blanks_score}",
            f"Post-Test Score: {self.post_score}",
            f"Learning gain: {self.learning_gain}",
            f"Relative learning gain: {self.relative_learning_gain}",
        ]
        lines = "\n".join([f"  {field}" for field in fields])

        return f"Answer(\n{lines}\n)"


def truncate(text: str, length: int) -> str:
    return text if len(text) <= length else text[:length] + "..."


def parse_datetime(date_str: str) -> datetime:
    return datetime.strptime(date_str, "%d/%m/%Y %H:%M:%S")


def parse_comma_separated(value: str) -> List[str]:
    return [item.strip() for item in value.split(",")]


def grade_words(should_have_checked: List[str], actual: List[str]) -> int:
     # points for words that were correctly checked
    points_for_matches = sum(1 for word in actual if word in should_have_checked)

    # points for words should not be checked, negative points
    points_for_missing = 8 - len(should_have_checked) + sum(-2 for word in should_have_checked if word not in actual)

    score = points_for_matches + points_for_missing
    return max(0, score)


def grade_blanks(expected: List[str], actual: List[str]) -> int:
    return sum([bool(expected[i] == actual[i]) for i in range(len(expected))])


def parse_survey_answers(csv_path: str, grading_path: str) -> List[Answer]:
    answers = []

    with open(grading_path, "r") as json_file:
        grading = json.load(json_file)

    with open(csv_path, "r") as csv_file:
        reader = csv.reader(csv_file)

        # Skip header row
        next(reader)

        missing_grading = False

        offset = 2
        for row_num, row in enumerate(reader, start=offset):
            row_num = row_num - offset

            # Skip empty rows (check if all values are empty)
            if all(not value.strip() for value in row):
                offset += 1
                continue

            if len(row) != 17:
                raise ValueError(f"Row {row_num}: Expected 17 columns, got {len(row)}")

            start_time_str = row[16].strip() if len(row) > 16 else ''
            end_time_str = row[0].strip()

            if not start_time_str:
                raise ValueError(f"Row {row_num}: no start time")

            try:
                end_time = parse_datetime(end_time_str)
                start_time = parse_datetime(start_time_str)
            except ValueError as e:
                raise ValueError(f"Row {row_num}: Invalid datetime format - {e}")

            duration = end_time - start_time

            activity_type = "I-PS" if row[9].strip() != "" else "PS-I"
            ps_link = row[9].strip() or row[10].strip()

            # words grades
            pre_dfa_words = parse_comma_separated(row[3].strip())
            post_dfa_words = parse_comma_separated(row[11].strip())
            pre_dfa_words_grade = grade_words(["bb", "bba", "bab", "abab", "baba"], pre_dfa_words)
            post_dfa_words_grade = grade_words(["aa", "aab", "aba"], post_dfa_words)

            # blanks grades
            pre_blanks = [row[5].strip(), row[6].strip(), row[7].strip()]
            post_blanks = [row[13].strip(), row[14].strip(), row[15].strip()]
            pre_blanks_grade = grade_blanks(["a", "a", "b"], pre_blanks)
            post_blanks_grade = grade_blanks(["a", "a", "b"], post_blanks)

            # property grades (open question, grade based on JSON file)
            pre_dfa_property = row[4].strip()
            post_dfa_property = row[12].strip()
            pre_dfa_property_grade = get_or_write(grading_path, grading, "pre_dfa_property_grade", pre_dfa_property)
            post_dfa_property_grade = get_or_write(grading_path, grading, "post_dfa_property_grade", post_dfa_property)

            if pre_dfa_property_grade is None or post_dfa_property_grade is None:
                missing_grading = True
                continue

            # scores
            pre_dfa_words_score = pre_dfa_words_grade / 8
            pre_dfa_property_score = pre_dfa_property_grade / 2
            pre_blanks_score = pre_blanks_grade / 3
            post_dfa_words_score = post_dfa_words_grade / 8
            post_dfa_property_score = post_dfa_property_grade / 2
            post_blanks_score = post_blanks_grade / 3

            answer = Answer(
                index=row_num,
                start_time=start_time,
                end_time=end_time,
                duration=duration,
                gender=row[1].strip(),
                age_group=row[2].strip(),
                pre_dfa_words=pre_dfa_words,
                pre_dfa_words_grade=pre_dfa_words_grade,
                pre_dfa_words_score=pre_dfa_words_score,
                pre_dfa_property=pre_dfa_property,
                pre_dfa_property_grade=pre_dfa_property_grade,
                pre_dfa_property_score=pre_dfa_property_score,
                pre_blanks=pre_blanks,
                pre_blanks_grade=pre_blanks_grade,
                pre_blanks_score=pre_blanks_score,
                pre_score=(pre_dfa_words_score + pre_dfa_property_score + pre_blanks_score) / 3,
                activity_type=activity_type,
                ps_link=ps_link,
                post_dfa_words=post_dfa_words,
                post_dfa_words_grade=post_dfa_words_grade,
                post_dfa_words_score=post_dfa_words_score,
                post_dfa_property=post_dfa_property,
                post_dfa_property_grade=post_dfa_property_grade,
                post_dfa_property_score=post_dfa_property_score,
                post_blanks=post_blanks,
                post_blanks_grade=post_blanks_grade,
                post_blanks_score=post_blanks_score,
                post_score=(post_dfa_words_score + post_dfa_property_score + post_blanks_score) / 3,
            )
            answers.append(answer)

        if missing_grading:
            raise ValueError(f"Missing some grading, fill the json.")

    return answers


def get_or_write(grading_path, grading, key, value):
    val = grading.get(key, {}).get(value)

    if val is None:
        grading[key][value] = None
        with open(grading_path, "w") as json_file:
            json.dump(grading, json_file, indent=2)

    return val


def aggregate(
    answers: Iterable[Answer],
    mapping: Callable[[Answer], U]=lambda l: l,
    fold: Callable[[List[U]], T]=lambda l: l,
) -> T:
    return fold([mapping(answer) for answer in answers])


def aggregate_by(
    answers: Iterable[Answer],
    key: Callable[[Answer], K],
    mapping: Callable[[Answer], U]=lambda l: l,
    fold: Callable[[List[U]], T]=lambda l: l,
) -> Dict[K, T]:
    grouped: Dict[K, List[U]] = defaultdict(list)

    for answer in answers:
        grouped[key(answer)].append(mapping(answer))

    return {key: fold(values) for key, values in grouped.items()}


def select(
    answers: Iterable[Answer],
    test: Callable[[Answer], bool],
) -> List[Answer]:
    return list(filter(test, answers))


def main():
    # utils
    mean = lambda l: sum(l) / len(l)
    activity_type = lambda a: a.activity_type
    pre_score = lambda a: a.pre_score
    post_score = lambda a: a.post_score
    rel_learning_gain = lambda a: a.relative_learning_gain

    answers = parse_survey_answers(SURVEY_CSV, GRADING_JSON)

    selected_answers = select(answers, lambda a: a.duration >= timedelta(minutes=25))

    print(f"Total answers count: {len(answers)}")
    print(f"Total answers groups: {aggregate_by(answers, activity_type, fold=len)}")
    print()

    # print(f"Pre-test scores: {group(answers, pre_score)}")
    print(f"Pre-test mean: {aggregate(answers, pre_score, mean)}")
    print()

    print(f"Pre-test means: {aggregate_by(answers, activity_type, pre_score, mean)}")
    print(f"Post-test means: {aggregate_by(answers, activity_type, post_score, mean)}")
    print(f"Selected post-test means: {aggregate_by(selected_answers, activity_type, post_score, mean)}")
    print()

    print(f"Relative learning gain mean: {aggregate(answers, rel_learning_gain, mean)}")
    print(f"Relative learning gain means: {aggregate_by(answers, activity_type, rel_learning_gain, mean)}")
    print(f"Selected relative learning gain means: {aggregate_by(selected_answers, activity_type, rel_learning_gain, mean)}")
    print()

    # for answer in answers:
    #     print(answer)

    # print(answers[-1])

    # matplotlib boxplot
    # statsmodels.api sm.stats.anova_lm


if __name__ == "__main__":
    main()
