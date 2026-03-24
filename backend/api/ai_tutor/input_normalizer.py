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
            
            Combine the text and images above into a single self-contained problem statement.
            
            Rules:
            - If the text is already complete and no images add info → keep original wording
            - If images contain text → extract exactly
            - If images contain a diagram, figure, or visual → describe EVERY labeled value,
            measurement, marking, connection, and spatial relationship so someone can
            understand the problem without seeing the image
            - If images show a real-world object, scene, or item → describe what is
            visible clearly and concisely
            - Merge the text and image descriptions into one coherent problem statement
            - The output must make full sense on its own without the original image
            - Return ONLY the final problem statement — no preamble, no commentary
          
            """
        ]
    )

    return response.text