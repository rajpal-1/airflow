#
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""DAG demonstrating various options for a trigger form generated by DAG params.

The DAG attribute `params` is used to define a default dictionary of parameters which are usually passed
to the DAG and which are used to render a trigger form.
"""
from __future__ import annotations

import datetime
import json
from pathlib import Path

from airflow.decorators import task
from airflow.models.dag import DAG
from airflow.models.param import Param, ParamsDict

with DAG(
    dag_id=Path(__file__).stem,
    description=__doc__.partition(".")[0],
    doc_md=__doc__,
    schedule=None,
    start_date=datetime.datetime(2022, 3, 4),
    catchup=False,
    tags=["example_ui"],
    params={
        # Let's start simple: Standard dict values are detected from type and offered as entry form fields.
        # Detected types are numbers, text, boolean, lists and dicts.
        # Note that such auto-detected parameters are treated as optional (not required to contain a value)
        "x": 3,
        "text": "Hello World!",
        "flag": False,
        "a_simple_list": ["one", "two", "three", "actually one value is made per line"],
        # But of course you might want to have it nicer! Let's add some description to parameters.
        # Note if you can add any MD formatting to the description, you need to use the description_md
        # attribute.
        "most_loved_number": Param(
            42,
            type="integer",
            title="Your favorite number",
            description_md="Everybody should have a **favorite** number. Not only _math teachers_. "
            "If you can not think of any at the moment please think of the 42 which is very famous because"
            "of the book [The Hitchhiker's Guide to the Galaxy]"
            "(https://en.wikipedia.org/wiki/Phrases_from_The_Hitchhiker%27s_Guide_to_the_Galaxy#"
            "The_Answer_to_the_Ultimate_Question_of_Life,_the_Universe,_and_Everything_is_42).",
        ),
        # If you want to have a selection list box then you can use the enum feature of JSON schema
        "pick_one": Param(
            "value 42",
            type="string",
            title="Select one Value",
            description="You can use JSON schema enum's to generate drop down selection boxes.",
            enum=[f"value {i}" for i in range(16, 64)],
        ),
        # You can also label the selected values via values_display attribute
        "pick_with_label": Param(
            3,
            type="number",
            title="Select one Number",
            description="With drop down selections you can also have nice display labels for the values.",
            enum=[*range(1, 10)],
            values_display={
                1: "One",
                2: "Two",
                3: "Three",
                4: "Four - is like you take three and get one for free!",
                5: "Five",
                6: "Six",
                7: "Seven",
                8: "Eight",
                9: "Nine",
            },
        ),
        # If you want to have a list box with proposals but not enforcing a fixed list
        # then you can use the examples feature of JSON schema
        "proposals": Param(
            "some value",
            type="string",
            title="Field with proposals",
            description="You can use JSON schema examples's to generate drop down selection boxes "
            "but allow also to enter custom values. Try typing an 'a' and see options.",
            examples=(
                "Alpha,Bravo,Charlie,Delta,Echo,Foxtrot,Golf,Hotel,India,Juliett,Kilo,Lima,Mike,November,Oscar,Papa,"
                "Quebec,Romeo,Sierra,Tango,Uniform,Victor,Whiskey,X-ray,Yankee,Zulu"
            ).split(","),
        ),
        # If you want to select multiple items from a fixed list JSON schema des not allow to use enum
        # In this case the type "array" is being used together with "examples" as pick list
        "multi_select": Param(
            ["two", "three"],
            "Select from the list of options.",
            type="array",
            title="Multi Select",
            examples=["one", "two", "three", "four", "five"],
        ),
        # A multiple options selection can also be combined with values_display
        "multi_select_with_label": Param(
            ["2", "3"],
            "Select from the list of options. See that options can have nicer text and still technical values"
            "are propagated as values during trigger to the DAG.",
            type="array",
            title="Multi Select with Labels",
            examples=["1", "2", "3", "4", "5"],
            values_display={
                "1": "One box of choccolate",
                "2": "Two bananas",
                "3": "Three apples",
                # Note: Value display mapping does not need to be complete.s
            },
        ),
        # An array of numbers
        "array_of_numbers": Param(
            [1, 2, 3],
            "Only integers are accepted in this array",
            type="array",
            title="Array of numbers",
            items={"type": "number"},
        ),
        # Boolean as proper parameter with description
        "bool": Param(
            True,
            type="boolean",
            title="Please confirm",
            description="A On/Off selection with a proper description.",
        ),
        # Dates and Times are also supported
        "date_time": Param(
            f"{datetime.date.today()}T{datetime.time(hour=12, minute=17, second=00)}+00:00",
            type="string",
            format="date-time",
            title="Date-Time Picker",
            description="Please select a date and time, use the button on the left for a pup-up calendar.",
        ),
        "date": Param(
            f"{datetime.date.today()}",
            type="string",
            format="date",
            title="Date Picker",
            description="Please select a date, use the button on the left for a pup-up calendar. "
            "See that here are no times!",
        ),
        "time": Param(
            f"{datetime.time(hour=12, minute=13, second=14)}",
            type=["string", "null"],
            format="time",
            title="Time Picker",
            description="Please select a time, use the button on the left for a pup-up tool.",
        ),
        # Fields can be required or not. If the defined fields are typed they are getting required by default
        # (else they would not pass JSON schema validation) - to make typed fields optional you must
        # permit the optional "null" type.
        # You can omit a default value if the DAG is triggered manually
        "required_field": Param(
            # In this example we have no default value
            # Form will enforce a value supplied by users to be able to trigger
            type="string",
            title="Required text field",
            description="This field is required. You can not submit without having text in here.",
        ),
        "optional_field": Param(
            "optional text, you can trigger also w/o text",
            type=["null", "string"],
            title="Optional text field",
            description_md="This field is optional. As field content is JSON schema validated you must "
            "allow the `null` type.",
        ),
        # You can arrange the entry fields in sections so that you can have a better overview for the user
        # Therefore you can add the "section" attribute.
        # The benefit of the Params class definition is that the full scope of JSON schema validation
        # can be leveraged for form fields and they will be validated before DAG submission.
        "checked_text": Param(
            "length-checked-field",
            type="string",
            title="Text field with length check",
            description_md="""This field is required. And you need to provide something between 10 and 30
            characters. See the JSON
            [schema description (string)](https://json-schema.org/understanding-json-schema/reference/string.html)
            for more details""",
            minLength=10,
            maxLength=20,
            section="JSON Schema validation options",
        ),
        "checked_number": Param(
            100,
            type="number",
            title="Number field with value check",
            description_md="""This field is required. You need to provide any number between 64 and 128.
            See the JSON
            [schema description (numbers)](https://json-schema.org/understanding-json-schema/reference/numeric.html)
            for more details""",
            minimum=64,
            maximum=128,
            section="JSON Schema validation options",
        ),
        # Some further cool stuff as advanced options are also possible
        # You can have the user entering a dict object as a JSON with validation
        "object": Param(
            {"key": "value"},
            type=["object", "null"],
            title="JSON entry field",
            section="Special advanced stuff with form fields",
        ),
        "array_of_objects": Param(
            [{"name": "account_name", "country": "country_name"}],
            description_md="Array with complex objects and validation rules. "
            "See [JSON Schema validation options in specs]"
            "(https://json-schema.org/understanding-json-schema/reference/array.html#items).",
            type="array",
            title="JSON array field",
            items={
                "type": "object",
                "properties": {"name": {"type": "string"}, "country_name": {"type": "string"}},
                "required": ["name"],
            },
            section="Special advanced stuff with form fields",
        ),
        # If you want to have static parameters which are always passed and not editable by the user
        # then you can use the JSON schema option of passing constant values. These parameters
        # will not be displayed but passed to the DAG
        "hidden_secret_field": Param("constant value", const="constant value"),
    },
) as dag:

    @task
    def show_params(**kwargs) -> None:
        params: ParamsDict = kwargs["params"]
        print(f"This DAG was triggered with the following parameters:\n\n{json.dumps(params, indent=4)}\n")

    show_params()
