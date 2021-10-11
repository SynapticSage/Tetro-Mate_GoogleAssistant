
def get_notes(tet, DF=None):
    if DF is None:
        DF = EL.raw2pretty()[0]
    return DF[tet][['turns','notes','depth_mm']].set_index('depth_mm',append=True)
