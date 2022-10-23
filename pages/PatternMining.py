from PAMI.frequentPattern.basic import FPGrowth as freq_alg
from PAMI.frequentPattern.basic import Apriori as freq_apriori
from PAMI.frequentPattern.closed import CHARM as closed_alg
from PAMI.frequentPattern.maximal import MaxFPGrowth as maximal_alg
from PAMI.frequentPattern.topk import FAE as topK_alg

# from PAMI.sequentialPatternMining import prefixSpan as seq_alg
# from PAMI.AssociationRules import RuleMiner as rule_alg
import streamlit as st


tab1, tab2, tab3, tab4 = st.tabs(["Find frequent patterns", "Find Closed Pattern", "Find Maximal Pattern", "Find top-k Patterns"])

with tab1:
    # Initialize the FP-growth algorithm by providing the file, minimum support (minSup), and separator as the input parameters.
    obj = freq_alg.FPGrowth(st.session_state.uploaded_file ,100,'\t')
    # Start mining the pattern
    obj.startMine()
    # Show the discovered patterns as pandas DataFrame
    df = obj.getPatternsAsDataFrame()
    st.write('The discovered patterns is shown below:')
    st.write(df)


with tab2:
    obj = closed_alg.CHARM(st.session_state.uploaded_file, 100,'\t')
    obj.startMine()
    # obj.savePatterns('closedPatters_100.txt')
    df2 = obj.getPatternsAsDataFrame()
    st.write('The discovered patterns is shown below:')
    st.write(df2)

with tab3:
    obj = maximal_alg.MaxFPGrowth(st.session_state.uploaded_file,100,'\t')
    obj.startMine()
    #obj.savePatterns('maximalPatters_100.txt')
    df = obj.getPatternsAsDataFrame()
    st.write('The discovered patterns is shown below:')
    st.write(df)

with tab4:
    obj = topK_alg.FAE(st.session_state.uploaded_file,10,'\t')
    obj.startMine()
    #obj.savePatterns('topKPatters_100.txt')
    df = obj.getPatternsAsDataFrame()
    st.write('The discovered patterns is shown below:')
    st.write(df)


