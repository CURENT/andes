import modparc
model_definition = modparc.parse_file("AVRtypeIII.mo")

all_comments = model_definition.search('EquationSection')

for comment in all_comments:
    print(comment.code())