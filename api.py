from huggingface_hub import InferenceClient
from pydantic import BaseModel
# from openai import OpenAI

openai_compatible = True

client = InferenceClient(
    model="https://llm.vse.cz/tgi/v1" if openai_compatible
            else "https://llm.vse.cz/tgi"
)

class Model_Reponse_Type(BaseModel):
    answer: bool

model_response_schema = Model_Reponse_Type.schema()

response = client.chat.completions.create(
    messages=[{"role": "system", "content": "Only answer with 'yes' to any question.",
            "role": "user", "content": """I give you a small CSV. The task for you (IT IS YOUR TASK TO DO) is to find frequent itemsets with maxlen = 5 and support = 27/54.
<IMPORTANT>Do NOT use programming languages!!!</IMPORTANT>
The CSV is provided below:

<CSV>
id,age_gt_60,air,airBoneGap,ar_c,ar_u,bone,boneAbnormal,bser,history_buzzing,history_dizziness,history_fluctuating,history_fullness,history_heredity,history_nausea,history_noise,history_recruitment,history_ringing,history_roaring,history_vomiting,late_wave_poor,m_at_2k,m_cond_lt_1k,m_gt_1k,m_m_gt_2k,m_m_sn,m_m_sn_gt_1k,m_m_sn_gt_2k,m_m_sn_gt_500,m_p_sn_gt_2k,m_s_gt_500,m_s_sn,m_s_sn_gt_1k,m_s_sn_gt_2k,m_s_sn_gt_3k,m_s_sn_gt_4k,m_sn_2_3k,m_sn_gt_1k,m_sn_gt_2k,m_sn_gt_3k,m_sn_gt_4k,m_sn_gt_500,m_sn_gt_6k,m_sn_lt_1k,m_sn_lt_2k,m_sn_lt_3k,middle_wave_poor,mod_gt_4k,mod_mixed,mod_s_mixed,mod_s_sn_gt_500,mod_sn,mod_sn_gt_1k,mod_sn_gt_2k,mod_sn_gt_3k,mod_sn_gt_4k,mod_sn_gt_500,notch_4k,notch_at_4k,o_ar_c,o_ar_u,s_sn_gt_1k,s_sn_gt_2k,s_sn_gt_4k,speech,static_normal,tymp,viith_nerve_signs,wave_V_delayed,waveform_ItoV_prolonged,binaryClass
185,t,mild,f,normal,normal,?,f,?,f,f,f,f,f,f,t,f,f,f,f,f,f,f,f,f,f,t,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,normal,normal,f,f,f,very_poor,t,a,f,f,f,N
154,t,mild,f,normal,normal,mild,t,?,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,normal,normal,f,f,f,good,t,a,f,f,f,P
165,t,moderate,f,normal,normal,?,t,?,f,f,f,f,f,f,t,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,elevated,elevated,f,f,f,very_poor,t,a,f,f,f,N
24,f,moderate,t,absent,absent,mild,f,?,f,f,f,f,f,f,f,f,t,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,t,f,f,f,f,f,f,f,f,f,f,absent,normal,f,f,f,normal,t,as,f,f,f,N
2,t,mild,f,normal,normal,unmeasured,f,?,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,t,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,normal,normal,f,f,f,good,t,a,f,f,f,P
148,t,mild,f,normal,normal,?,t,?,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,normal,normal,f,f,f,good,t,a,f,f,f,P
220,t,mild,f,normal,normal,unmeasured,f,?,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,t,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,normal,elevated,f,f,f,good,t,a,f,f,f,P
47,t,mild,f,normal,normal,?,t,?,f,f,f,f,f,f,t,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,normal,normal,t,f,f,very_poor,t,a,f,f,f,N
48,t,mild,f,normal,normal,?,t,?,f,f,f,f,f,f,t,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,normal,normal,f,t,f,normal,t,a,f,f,f,N
7,t,mild,f,normal,normal,unmeasured,f,?,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,t,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,normal,normal,f,f,f,good,t,a,f,f,f,P
82,f,normal,f,elevated,elevated,normal,f,?,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,t,f,f,f,elevated,absent,f,f,f,normal,t,a,f,f,f,N
183,f,normal,f,normal,normal,normal,f,?,f,f,f,f,f,f,t,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,normal,normal,f,f,f,normal,t,a,f,f,f,N
9,f,normal,f,normal,normal,unmeasured,f,?,f,f,f,f,f,f,t,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,t,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,t,normal,normal,f,f,f,normal,t,a,f,f,f,N
118,f,mild,f,normal,normal,?,f,?,f,t,t,f,f,f,f,f,f,f,t,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,t,f,f,t,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,normal,normal,f,f,f,normal,f,a,f,f,f,N
5,t,mild,f,normal,normal,unmeasured,f,?,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,t,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,normal,normal,f,f,f,normal,t,a,f,f,f,P
20,t,mild,f,normal,normal,unmeasured,f,?,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,t,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,normal,normal,f,f,f,good,t,a,f,f,f,P
16,t,normal,f,normal,normal,unmeasured,f,?,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,t,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,normal,normal,f,f,f,good,t,a,f,f,f,P
13,f,normal,f,elevated,absent,normal,f,?,f,f,f,f,f,f,t,f,t,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,t,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,t,elevated,elevated,f,f,f,normal,t,a,f,f,f,N
163,f,normal,f,normal,normal,?,f,?,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,normal,normal,f,f,f,normal,t,a,f,f,f,N
103,f,mild,f,absent,absent,mild,f,?,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,absent,normal,f,f,f,normal,t,a,t,f,f,N
49,f,normal,f,normal,normal,?,t,?,f,f,f,f,f,f,t,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,t,f,normal,normal,f,t,f,good,t,a,f,f,f,N
38,t,mild,f,normal,normal,?,t,?,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,normal,normal,f,f,f,good,t,a,f,f,f,P
186,t,mild,f,normal,normal,?,f,?,f,f,f,f,f,f,t,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,t,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,t,f,normal,normal,f,f,f,?,t,a,f,f,f,N
161,f,severe,t,absent,absent,normal,f,?,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,absent,absent,f,f,f,poor,t,as,f,f,f,N
115,f,mild,f,elevated,normal,mild,f,?,f,t,t,f,f,f,f,f,f,t,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,normal,elevated,f,f,f,good,f,a,f,f,f,N
204,t,mild,f,normal,normal,unmeasured,f,?,f,f,f,f,f,f,t,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,t,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,normal,normal,f,f,f,very_good,t,a,f,f,f,P
96,f,mild,f,elevated,elevated,mild,f,?,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,t,f,f,elevated,normal,f,f,f,poor,t,a,f,f,f,N
193,f,moderate,t,absent,absent,mild,f,?,f,f,f,f,f,f,f,f,f,f,f,f,t,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,absent,absent,f,f,f,normal,t,as,f,f,f,N
26,t,normal,f,elevated,normal,unmeasured,f,?,f,f,f,f,f,f,t,f,f,f,f,f,f,f,f,f,f,f,f,f,t,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,t,elevated,elevated,f,f,f,very_poor,t,as,f,f,f,N
77,t,moderate,f,normal,elevated,?,f,?,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,normal,normal,f,t,f,very_poor,f,a,f,f,f,P
94,t,mild,f,absent,?,?,f,?,f,f,f,f,f,f,t,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,t,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,t,f,absent,absent,f,f,f,good,t,a,f,f,f,N
182,t,normal,f,normal,normal,?,f,?,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,t,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,normal,normal,f,f,f,normal,t,a,f,f,f,P
57,f,mild,f,normal,normal,?,f,?,f,f,f,f,f,f,t,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,t,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,t,f,elevated,normal,f,f,f,normal,t,a,f,f,f,N
175,f,mild,f,normal,normal,mild,f,?,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,t,f,f,f,f,f,f,f,f,f,f,f,f,f,normal,normal,f,f,f,normal,t,a,f,f,f,N
68,f,normal,f,normal,elevated,normal,f,?,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,absent,absent,f,f,f,normal,t,a,f,f,f,N
151,f,normal,f,normal,normal,normal,t,?,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,t,f,f,f,f,normal,elevated,f,f,f,very_good,t,a,f,f,f,N
200,f,normal,f,normal,normal,unmeasured,f,?,f,f,f,f,f,f,t,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,t,f,f,f,normal,normal,f,f,f,normal,t,a,f,f,f,N
171,t,mild,f,normal,normal,?,f,?,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,normal,normal,f,f,f,normal,t,a,f,f,f,P
149,t,mild,f,normal,elevated,mild,t,?,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,normal,elevated,f,f,f,good,t,a,f,f,f,P
109,f,moderate,t,absent,absent,mild,f,?,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,absent,absent,f,f,f,very_good,t,as,f,f,f,N
86,f,normal,f,normal,elevated,?,f,?,f,f,f,f,f,f,t,f,f,f,f,f,f,f,f,f,f,f,t,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,t,f,elevated,normal,f,f,f,very_good,t,a,f,f,f,N
56,f,mild,f,elevated,normal,mild,t,?,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,absent,absent,f,f,f,normal,t,a,f,f,f,N
158,t,normal,f,elevated,normal,?,f,?,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,t,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,normal,normal,f,f,f,good,t,a,f,f,f,P
33,f,mild,f,normal,normal,mild,t,?,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,normal,normal,f,f,f,good,t,a,f,f,f,N
91,f,normal,f,normal,normal,normal,f,?,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,elevated,elevated,f,f,f,normal,t,a,f,f,f,N
196,t,mild,f,normal,normal,mild,t,?,f,t,f,f,f,f,f,f,f,f,f,f,f,f,f,f,t,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,normal,elevated,f,f,f,good,t,a,f,f,f,P
136,f,mild,f,absent,absent,normal,f,?,f,t,f,f,f,f,t,f,t,f,f,f,f,f,t,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,absent,elevated,f,f,f,very_good,t,ad,f,f,f,N
119,f,normal,f,normal,normal,?,f,?,f,t,f,f,f,f,f,f,f,f,t,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,t,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,normal,normal,f,f,f,normal,f,a,f,f,f,N
150,f,mild,f,normal,normal,mild,t,?,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,normal,normal,f,f,f,normal,t,a,f,f,f,N
63,t,mild,f,absent,absent,mild,t,?,f,f,f,f,f,f,t,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,t,f,absent,absent,f,f,f,very_good,t,a,f,f,f,N
54,f,mild,f,normal,normal,?,t,?,f,f,f,f,f,f,t,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,normal,normal,f,f,t,normal,t,a,f,f,f,N
28,f,moderate,f,normal,normal,?,t,?,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,normal,normal,f,f,f,normal,t,a,f,f,f,N
157,f,mild,t,absent,absent,normal,f,?,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,absent,absent,f,f,f,?,t,as,f,f,f,N
120,f,normal,f,normal,normal,?,f,?,f,f,f,f,f,f,t,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,t,f,f,f,t,normal,normal,f,f,f,normal,t,a,f,f,f,N
</CSV>
"""}],
    temperature=1.0
) if openai_compatible else client.text_generation(
    prompt="Generate {{ 'answer': True }}",
    grammar=model_response_schema
)

print(response)
