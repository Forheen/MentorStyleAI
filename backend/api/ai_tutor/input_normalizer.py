import base64
from google.genai import types

from backend.core.gemini_client import client


def normalize_input(problem, images):

    # CASE 1 — Only text → return directly
    if problem and not images:

        return problem.strip()

    contents = []

    if problem:

        contents.append(problem)

    if images:

        for img in images:

            image_bytes = base64.b64decode(img)

            contents.append(

                types.Part.from_bytes(

                    data=image_bytes,

                    mime_type="image/png"

                )
            )

    if not contents:

        return None

    response = client.models.generate_content(

        model="gemini-2.0-flash",

        contents=[

            *contents,

            """
            Extract the full question from text and images.

            Rules:
            - If text already complete → keep original wording
            - If images contain text → extract exactly
            - If diagrams exist → describe clearly
            - Merge without rephrasing unnecessarily

            Return only the final question.
            """
        ]
    )

    return response.text