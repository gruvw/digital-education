import json
import csv
from typing import Callable, Dict, Iterable, List, TypeVar
from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import numpy as np
from scipy import stats
from statistics import median
from streamlit.runtime.caching import save_media_data

from ANOVA import multi_way_anova,shapiro_test,kruskal_one_way



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

    with open(csv_path, "r", encoding="utf-8") as csv_file:
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


def keep(
    answers: Iterable[Answer],
    test: Callable[[Answer], bool],
) -> List[Answer]:
    return list(filter(test, answers))


def bucket_genders(genders):
    c = Counter(gender for gender in genders)
    male = c.get("Male", 0)
    female = c.get("Female", 0)
    other = sum(v for k,v in c.items() if k not in ("Male","Female"))
    return [male, female, other]


def bucket_age_ranges(age_ranges, ranges):
    c = Counter(age_range.strip().lower() for age_range in age_ranges)
    return [c.get(group, 0) for group in ranges]

def safe_median(xs):
    cleaned = []
    for x in xs:
        if x is None:
            continue
        try:
            cleaned.append(float(x))
        except (TypeError, ValueError):
            continue
    return median(cleaned) if cleaned else None


def main():
    # utils
    mean = lambda l: sum(l) / len(l)
    activity_type = lambda a: a.activity_type
    pre_score = lambda a: a.pre_score
    post_score = lambda a: a.post_score
    learning_gain = lambda a: a.learning_gain
    rel_learning_gain = lambda a: a.relative_learning_gain
    duration = lambda a: a.duration
    age = lambda a: a.gender
    age_range = lambda a: a.age_group

    answers = parse_survey_answers(SURVEY_CSV, GRADING_JSON)

    # = Exclusion criteria =

    total_len = len(answers)
    selected_answers = keep(answers, lambda a: a.activity_type=="I-PS")

    print(f"Exclusion criteria excluded {total_len - len(answers)}")
    print(shapiro_test(selected_answers,"learning_gain"))

    # = General info =

    print(f"Total answers count: {len(answers)}")
    print(f"Total answers groups: {aggregate_by(answers, activity_type, fold=len)}")
    print()

    print(f"Mean time spent groups: {aggregate_by(answers, activity_type, duration, lambda l: (sum(l, timedelta(0)) / len(l)).total_seconds() / 60)}")
    print()

    # print(f"Pre-test scores: {group(answers, pre_score)}")
    print(f"Pre-test mean: {aggregate(answers, pre_score, mean)}")
    print()

    print(f"Pre-test means: {aggregate_by(answers, activity_type, pre_score, mean)}")
    print(f"Post-test means: {aggregate_by(answers, activity_type, post_score, mean)}")

    print(f"Pre-test medians: {aggregate_by(answers, activity_type, pre_score, safe_median)}")
    print(f"Post-test medians: {aggregate_by(answers, activity_type, post_score, safe_median)}")


    print()

    rel_learning_mean = aggregate(answers, rel_learning_gain, mean)
    print(f"Relative learning gain mean: {rel_learning_mean}")
    print(f"Relative learning gain means: {aggregate_by(answers, activity_type, rel_learning_gain, mean)}")
    print()


    print(f"learning gain means: {aggregate_by(answers, activity_type, learning_gain, mean)}")
    print()

    # for answer in answers:
    #     print(answer)

    # print(answers[-1])

    pre_test = aggregate_by(answers, activity_type, pre_score)
    post_test = aggregate_by(answers, activity_type, post_score)
    learning = aggregate_by(answers, activity_type, learning_gain)
    rel_learning = aggregate_by(answers, activity_type, rel_learning_gain)

    fontsize = 14
    titlesize = 16

    scale = [i / 5 for i in range(-5, 5 + 1)]

    # = ANOVA =

    print(multi_way_anova(answers,["activity_type"], "learning_gain")["anova_table"])
    print()
    print(kruskal_one_way(answers,"activity_type", "relative_learning_gain"))

    # = Demographics plot =

    genders = aggregate_by(answers, activity_type, age, bucket_genders)

    gender_colors = ["#4A90E2", "#FF6EB4", "#C0C0C0"] # Male, Female, Others

    fig, axs = plt.subplots(1,2, figsize=(8, 4))
    _, _, texts0 = axs[0].pie(
        genders["I-PS"],
        autopct=lambda p: "%1.1f%%" % p if p > 0 else "",
        startangle=90,
        colors=gender_colors,
        counterclock=False,
    )
    axs[0].set_title(f"I-PS ({sum(genders['I-PS'])})", fontsize=titlesize)
    axs[0].axis("equal")
    _, _, texts1 = axs[1].pie(
        genders["PS-I"],
        autopct=lambda p: "%1.1f%%" % p if p > 0 else "",
        startangle=90,
        colors=gender_colors,
        counterclock=False,
    )
    axs[1].set_title(f"PS-I ({sum(genders['PS-I'])})", fontsize=titlesize)
    axs[1].axis("equal")

    for text in [*texts0, *texts1]:
        text.set_fontsize(fontsize)

    fig.legend(
        [Patch(color=c) for c in gender_colors[:2]],
        ["Male", "Female"],
        loc="upper right",
        bbox_to_anchor=(1.15, 1),
        fontsize=fontsize,
    )

    fig.tight_layout()
    fig.savefig("genders_plot.png", dpi=150, bbox_inches="tight")
    # plt.show()
    plt.close()

    ranges = ["≤18", "19-21", "22-24", "25-27", "≥28"]
    age_ranges = aggregate_by(answers, activity_type, age_range, lambda l: bucket_age_ranges(l, ranges))

    age_colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974"]

    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    _, _, texts0 = axs[0].pie(
        age_ranges["I-PS"],
        colors=age_colors,
        autopct=lambda p: "%1.1f%%" % p if p > 0 else "",
        startangle=90,
        counterclock=False,
    )
    axs[0].set_title(f"I-PS ({sum(age_ranges['I-PS'])})", fontsize=titlesize)
    axs[0].axis("equal")

    _, _, texts1 = axs[1].pie(
        age_ranges["PS-I"],
        colors=age_colors,
        autopct=lambda p: "%1.1f%%" % p if p > 0 else "",
        startangle=90,
        counterclock=False,
    )
    axs[1].set_title(f"PS-I ({sum(age_ranges['PS-I'])})", fontsize=titlesize)
    axs[1].axis("equal")

    for text in [*texts0, *texts1]:
        text.set_fontsize(fontsize)

    fig.legend(
        [Patch(color=c) for c in age_colors],
        ranges,
        loc="upper right",
        bbox_to_anchor=(1.15, 1),
        fontsize=fontsize,
    )

    fig.tight_layout()
    fig.savefig("age_ranges_plot.png", dpi=150, bbox_inches="tight")
    # plt.show()
    plt.close()

    # = Learning gain plot =

    dataset = [
        pre_test["I-PS"], pre_test["PS-I"],
        post_test["I-PS"], post_test["PS-I"],
        learning["I-PS"], learning["PS-I"],
        rel_learning["I-PS"], rel_learning["PS-I"],
    ]

    box_width = 0.3
    widths = [box_width] * len(dataset)
    inter_delta = box_width + 0.1
    exter_delta = box_width + 0.4
    positions = [
        1, 1+inter_delta,
        1+inter_delta+exter_delta, 1+2*inter_delta+exter_delta,
        1+2*inter_delta+2*exter_delta, 1+3*inter_delta+2*exter_delta,
        1+3*inter_delta+3*exter_delta, 1+4*inter_delta+3*exter_delta,
    ]
    ips_color = "cornflowerblue"
    psi_color = "mediumseagreen"
    colors = [ips_color, psi_color]*len(dataset)

    meanprops = dict(
        marker="D",
        markeredgecolor="black",
        markerfacecolor="black",
        markersize=6,
    )

    plt.subplots(figsize=(7, 7.5))
    bp = plt.boxplot(
        dataset,
        positions=positions,
        patch_artist=True,
        medianprops=dict(color="black", linewidth=3),
        showmeans=True,
        widths=widths,
        meanprops=meanprops,
        showfliers=False,
    )
    for patch, color in zip(bp["boxes"], colors):
        patch.set(facecolor=color)

    plt.xticks(
        [(a + b) / 2 for a, b in zip(positions[::2], positions[1::2])],
        ["Pre-test\nscore", "Post-test\nscore", "Learning\ngain", "Relative learning\ngain"],
        fontsize=fontsize,
    )
    plt.xlim(positions[0]-exter_delta/2, positions[-1]+exter_delta/2)
    plt.yticks(fontsize=fontsize)
    plt.ylabel("Score", fontsize=titlesize)

    legend_patches = [
        Patch(facecolor=colors[0], edgecolor="none", label="I-PS"),
        Patch(facecolor=colors[1], edgecolor="none", label="PS-I"),
        Line2D([0], [0], color="w", label="Mean", **meanprops)
    ]
    plt.legend(
        handles=legend_patches,
        loc="upper left",
        fontsize=fontsize,
    )

    plt.grid(True, alpha=0.3, linestyle="-")
    plt.tight_layout()
    plt.savefig("learning_gain_plot.png", dpi=150)
    # plt.show()
    plt.close()

    # = Interaction plot (gender) =

    gender_data = {
        "Male": aggregate_by(keep(answers, lambda a: a.gender == "Male"), activity_type, rel_learning_gain),
        "Female": aggregate_by(keep(answers, lambda a: a.gender == "Female"), activity_type, rel_learning_gain),
    }

    genders = gender_data.keys()
    conditions = ["I-PS", "PS-I"]
    colors = [ips_color, psi_color]

    fig, ax = plt.subplots(figsize=(6, 6))

    x_positions = np.arange(len(genders))

    offset = 0.04
    n_boot = 5000 # bootstrap resamples
    confidence = 95
    alpha = (100 - confidence) / 2

    for i, (condition, color) in enumerate(zip(conditions, colors)):
        means = []
        ci_lower = []
        ci_upper = []

        for gender in genders:
            data = np.array(gender_data[gender][condition])
            mean = data.mean()
            means.append(mean)

            # bootstrap CI of the mean
            boot_means = []
            for _ in range(n_boot):
                sample = np.random.choice(data, size=len(data), replace=True)
                boot_means.append(sample.mean())
            boot_means = np.array(boot_means)

            lower = np.percentile(boot_means, alpha)
            upper = np.percentile(boot_means, 100 - alpha)

            ci_lower.append(mean - lower)
            ci_upper.append(upper - mean)

        x_offset = x_positions + (offset if i == 1 else -offset)
        ax.plot(x_offset, means, marker="D", markersize=8, color=color, linewidth=2, label=condition)
        ax.errorbar(x_offset, means, yerr=[ci_lower, ci_upper], fmt="none", ecolor=color, capsize=5, capthick=2, linewidth=2)

    ax.set_ylabel("Relative learning gain", fontsize=titlesize)
    ax.set_yticks(scale)
    ax.set_yticklabels(scale, fontsize=fontsize)
    ax.set_ylim(-0.2, 1.05)
    ax.set_xticks(x_positions)
    ax.set_xlabel("Gender", fontsize=titlesize)
    ax.set_xticklabels([f"{x}\n({len(gender_data[x][conditions[0]])}, {len(gender_data[x][conditions[1]])})" for x in genders], fontsize=fontsize)
    ax.legend(fontsize=fontsize, loc="lower center")
    ax.grid(True, alpha=0.3, linestyle="-")

    plt.tight_layout()
    plt.savefig("gender_interaction_plot.png", dpi=150, bbox_inches="tight")
    # plt.show()
    plt.close()

    # = Interaction plot (age) =

    age_data = {
        "<25": aggregate_by(keep(answers, lambda a: a.age_group == "19-21" or a.age_group == "22-24"), activity_type, rel_learning_gain),
        "≥25": aggregate_by(keep(answers, lambda a: a.age_group == "25-27" or a.age_group == "≥28"), activity_type, rel_learning_gain),
    }

    ages = age_data.keys()
    conditions = ["I-PS", "PS-I"]
    colors = [ips_color, psi_color]

    fig, ax = plt.subplots(figsize=(6, 6))

    x_positions = np.arange(len(ages))

    for i, (condition, color) in enumerate(zip(conditions, colors)):
        means = []
        ci_lower = []
        ci_upper = []

        for age in ages:
            data = np.array(age_data[age][condition])
            mean = data.mean()
            means.append(mean)

            # bootstrap CI of the mean
            boot_means = []
            for _ in range(n_boot):
                sample = np.random.choice(data, size=len(data), replace=True)
                boot_means.append(sample.mean())
            boot_means = np.array(boot_means)

            lower = np.percentile(boot_means, alpha)
            upper = np.percentile(boot_means, 100 - alpha)

            ci_lower.append(mean - lower)
            ci_upper.append(upper - mean)

        x_offset = x_positions + (offset if i == 1 else -offset)
        ax.plot(x_offset, means, marker="D", markersize=8, color=color, linewidth=2, label=condition)
        ax.errorbar(x_offset, means, yerr=[ci_lower, ci_upper], fmt="none", ecolor=color, capsize=5, capthick=2, linewidth=2)

    ax.set_ylabel("Relative learning gain", fontsize=titlesize)
    ax.set_yticks(scale)
    ax.set_yticklabels(scale, fontsize=fontsize)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("Age [years]", fontsize=titlesize)
    ax.set_xticks(x_positions)
    ax.set_xticklabels([f"{x}\n({len(age_data[x][conditions[0]])}, {len(age_data[x][conditions[1]])})" for x in ages], fontsize=fontsize)
    ax.legend(fontsize=fontsize, loc="lower center")
    ax.grid(True, alpha=0.3, linestyle="-")

    plt.tight_layout()
    plt.savefig("age_interaction_plot.png", dpi=150, bbox_inches="tight")
    # plt.show()
    plt.close()

    # = Interaction plot (duration) =

    duration_data = {
        "<35": aggregate_by(keep(answers, lambda a: a.duration.total_seconds() / 60 < 35), activity_type, rel_learning_gain),
        "35-50": aggregate_by(keep(answers, lambda a: 35 <= a.duration.total_seconds() / 60 <= 50), activity_type, rel_learning_gain),
        ">50": aggregate_by(keep(answers, lambda a: 50 < a.duration.total_seconds() / 60), activity_type, rel_learning_gain),
    }

    durations = duration_data.keys()
    conditions = ["I-PS", "PS-I"]
    colors = [ips_color, psi_color]

    fig, ax = plt.subplots(figsize=(6, 6))

    x_positions = np.arange(len(durations))

    for i, (condition, color) in enumerate(zip(conditions, colors)):
        means = []
        ci_lower = []
        ci_upper = []

        for duration in durations:
            data = np.array(duration_data[duration][condition])
            mean = data.mean()
            means.append(mean)

            # bootstrap CI of the mean
            boot_means = []
            for _ in range(n_boot):
                sample = np.random.choice(data, size=len(data), replace=True)
                boot_means.append(sample.mean())
            boot_means = np.array(boot_means)

            lower = np.percentile(boot_means, alpha)
            upper = np.percentile(boot_means, 100 - alpha)

            ci_lower.append(mean - lower)
            ci_upper.append(upper - mean)

        x_offset = x_positions + (offset if i == 1 else -offset)
        ax.plot(x_offset, means, marker="D", markersize=8, color=color, linewidth=2, label=condition)
        ax.errorbar(x_offset, means, yerr=[ci_lower, ci_upper], fmt="none", ecolor=color, capsize=5, capthick=2, linewidth=2)

    ax.set_ylabel("Relative learning gain", fontsize=titlesize)
    ax.set_yticks(scale)
    ax.set_yticklabels(scale, fontsize=fontsize)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("Duration [minutes]", fontsize=titlesize)
    ax.set_xticks(x_positions)
    ax.set_xticklabels([f"{x}\n({len(duration_data[x][conditions[0]])}, {len(duration_data[x][conditions[1]])})" for x in durations], fontsize=fontsize)
    ax.legend(fontsize=fontsize, loc="lower center")
    ax.grid(True, alpha=0.3, linestyle="-")

    plt.tight_layout()
    plt.savefig("duration_interaction_plot.png", dpi=150, bbox_inches="tight")
    # plt.show()
    plt.close()

    # == Power analysis ==

    # Extract data for the two conditions
    ips_gains = rel_learning["I-PS"]
    psi_gains = rel_learning["PS-I"]

    mean_ips = np.mean(ips_gains)
    mean_psi = np.mean(psi_gains)

    var_ips = np.var(ips_gains, ddof=1)
    var_psi = np.var(psi_gains, ddof=1)
    n_ips = len(ips_gains)
    n_psi = len(psi_gains)

    pooled_sd = np.sqrt(((n_ips - 1) * var_ips + (n_psi - 1) * var_psi) / (n_ips + n_psi - 2))
    cohens_f = abs(mean_ips - mean_psi) / (pooled_sd * np.sqrt(2))

    grand_mean = (n_ips * mean_ips + n_psi * mean_psi) / (n_ips + n_psi)
    ss_between = n_ips * (mean_ips - grand_mean)**2 + n_psi * (mean_psi - grand_mean)**2
    ss_within = (n_ips - 1) * var_ips + (n_psi - 1) * var_psi

    # Calculate F-statistic for ANOVA
    df_between = 1  # k - 1 where k = 2 groups
    df_within = n_ips + n_psi - 2
    ms_between = ss_between / df_between
    ms_within = ss_within / df_within
    f_statistic = ms_between / ms_within

    noncentrality = cohens_f**2 * (n_ips + n_psi)
    critical_f = stats.f.ppf(0.95, df_between, df_within)
    power_est = 1 - stats.ncf.cdf(critical_f, df_between, df_within, noncentrality)

    fig, ax = plt.subplots(figsize=(10, 6))

    f_max = max(critical_f * 2, 10)
    f_values = np.linspace(0, f_max, 1000)
    pdf_central = stats.f.pdf(f_values, df_between, df_within)
    pdf_noncentral = stats.ncf.pdf(f_values, df_between, df_within, noncentrality)

    ax.plot(f_values, pdf_central, "b-", linewidth=2, label="Null hypothesis (H₀)")
    ax.plot(f_values, pdf_noncentral, "r--", linewidth=2, label="Alternative hypothesis (H₁)")
    ax.axvline(critical_f, color="green", linewidth=2, linestyle='-', label=f"Critical F = {critical_f:.3f}")

    f_alpha = f_values[f_values >= critical_f]
    pdf_alpha = stats.f.pdf(f_alpha, df_between, df_within)
    ax.fill_between(f_alpha, 0, pdf_alpha, alpha=0.3, color="blue", label=f"α = {0.05:.2f} (Type I error)")

    f_beta = f_values[f_values <= critical_f]
    pdf_beta = stats.ncf.pdf(f_beta, df_between, df_within, noncentrality)
    ax.fill_between(f_beta, 0, pdf_beta, alpha=0.3, color="orange", label=f"β = {1-power_est:.3f} (Type II error)")

    f_power = f_values[f_values >= critical_f]
    pdf_power = stats.ncf.pdf(f_power, df_between, df_within, noncentrality)
    ax.fill_between(f_power, 0, pdf_power, alpha=0.5, color="red",
                     label=f"Power (1-β) = {power_est:.3f}")

    ax.axvline(f_statistic, color="black", linewidth=1.5, linestyle=":", label=f"Observed F = {f_statistic:.3f}")

    ax.set_xlabel("F-statistic", fontsize=titlesize)
    ax.set_ylabel("Probability Density", fontsize=titlesize)
    ax.set_title(f"Power Analysis: ANOVA (df₁={df_between}, df₂={df_within}, f={cohens_f:.3f}, n={n_ips + n_psi})", fontsize=titlesize)
    ax.legend(fontsize=fontsize, loc="upper right")
    ax.grid(True, alpha=0.3, linestyle="-")

    plt.tight_layout()
    plt.savefig("power_analysis_plot.png", dpi=300, bbox_inches="tight")
    # plt.show()
    plt.close()


if __name__ == "__main__":
    np.random.seed(42)

    main()
