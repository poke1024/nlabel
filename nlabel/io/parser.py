import parsimonious
from .common import Name


class TagNameVisitor(parsimonious.NodeVisitor):
    def visit_expr(self, node, visited_children):
        name, rename, typing = visited_children
        pub_name =rename[0].text if isinstance(rename, list) else None

        return {
            'name': Name(name.text, pub_name),
            'label_type': typing[0].text if isinstance(typing, list) else 'str'
        }

    def visit_rename(self, node, visited_children):
        return visited_children[-1]

    def visit_typing(self, node, visited_children):
        return visited_children[-1]

    def generic_visit(self, node, visited_children):
        return visited_children or node


class TagNameParser:
    def __init__(self):
        self._grammar = parsimonious.grammar.Grammar(
            r"""
                expr   = name rename? typing?
                rename = _ "as" _ name
                typing = _ ":" _ name
                name   = ~r"[-\w]+"
                _      = ~"\s*"
            """)

    def __call__(self, spec):
        try:
            tree = self._grammar.parse(spec)
            iv = TagNameVisitor()
            return iv.visit(tree)
        except parsimonious.ParseError:
            raise ValueError(
                f"'{spec}' is not a valid tag specification")
