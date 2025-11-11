import json


GRADING_JSON = "grading.json"


def three_int_range_grades(grading_path, grading, key):
    for answer, grade in grading.get(key, {}).items():
        if grade is None:
            grading[key][answer] = int(input(f"'{answer}' [0/1/2]: "))
            with open(grading_path, "w") as json_file:
                json.dump(grading, json_file, indent=2)


def main():
    with open(GRADING_JSON, "r") as json_file:
        grading = json.load(json_file)

    print("Pre-Test Grading:")
    print("CORRECT ANSWER: does not containt substring ba")

    three_int_range_grades(GRADING_JSON, grading, "pre_dfa_property_grade")

    print("Post-Test Grading:")
    print("CORRECT ANSWER: odd number of a")

    three_int_range_grades(GRADING_JSON, grading, "post_dfa_property_grade")


if __name__ == "__main__":
    main()
