
from openai import OpenAI
import os

INSTRUCTIONS = """
Your task is to answer questions from the course participants
based on the provided context.

Use the context to find relevant information and provide accurate
answers. If the answer is not found in the context,
respond with "I don't know."
"""

PROMPT_TEMPLATE = """
QUESTION: {question}

CONTEXT:
{context}
""".strip()


class RAGBase:
    def __init__(
        self, 
        course="llm-zoomcamp",
        PROMPT_TEMPLATE=PROMPT_TEMPLATE,
        INSTRUCTIONS=INSTRUCTIONS
        ):
        self.course = course
        self.prompt_template = PROMPT_TEMPLATE
        self.instructions = INSTRUCTIONS

    def search(self, question):
        raise NotImplementedError

    def _build_context(self, search_result):
        lines = []

        for doc in search_result:
            lines.append(doc["section"])
            lines.append("Q: " + doc["question"])
            lines.append("A: " + doc["answer"])
            lines.append("")

        return "\n".join(lines).strip()

    def _build_prompt(self, question):
        search_result = self.search(question)
        context = self._build_context(search_result)

        prompt = self.prompt_template.format(
            question=question,
            context=context
        )
        return prompt

    def rag(self, question):
        prompt = self._build_prompt(question)
        answer = self._llm(prompt)

        return answer

# Vectore Search
class RAGVectore(RAGBase):
    def __init__(self, vindex, embedder, **kwargs):
        super().__init__(**kwargs)
        self.vindex = vindex
        self.embedder = embedder

    def search(self, question):
        filter_dict = {"course": "llm-zoomcamp"}

        search_result = \
            self.vindex.search(
                self.embedder.encode(question),
                filter_dict=filter_dict,
                num_results=5
            )

        return search_result


from ingest import vec_to_str

class RAGPgVector(RAGBase):
    def __init__(self, embedder, conn, **kwargs):
        super().__init__(**kwargs)
        self.embedder = embedder
        self.conn = conn

    def search(self, question):
        query_vector = self.embedder.encode(question)
        query_str = vec_to_str(query_vector)

        rows = self.conn.execute(
            """
            SELECT course, section, question, answer
            FROM documents
            WHERE course = %s
            ORDER BY embedding <=> %s::vector
            LIMIT 5
            """,
            (self.course, query_str)
        ).fetchall()

        return [
            {
                "course": row[0],
                "section": row[1],
                "question": row[2],
                "answer": row[3]
            }
            for row in rows
        ]

class OpenAIRAG(RAGPgVector):
    def __init__(self, embedder, conn, model='gpt-5.4-mini'):
        super().__init__(embedder, conn)
        self.model = model
        self.llm_client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
        )

    def _llm(self, prompt):
        input_messages = [
            {"role": "developer", "content": self.instructions},
            {"role": "user", "content": prompt}
        ]

        response = self.llm_client.responses.create(
            model=self.model,
            input=input_messages
        )

        return response.output_text
