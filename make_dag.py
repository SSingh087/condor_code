#!/usr/bin/env python

population = "A"
work_dir = "/data/wiay/postgrads/shashwat/EMRI_data/SF_DATA_MODEL"
num_jobs = 100000
max_jobs_running = 20

dag_filename = f"sf_generation_{population}.dag"

file_content = ""
parent_list = []

for i in range(num_jobs):
    job_name = f"{population}_{i}"
    parent_list.append(job_name)

    file_content += f"""JOB {job_name} data_prep.sub
RETRY {job_name} 0
VARS {job_name} ID="{i}" POP="{population}"
CATEGORY {job_name} {population}

"""


# Add combine job block (preserved formatting)
combine_job = f"COMBINE_{population}"
file_content += f"""JOB {combine_job} combine_SF.sub
VARS {combine_job} POP="{population}"

"""

file_content += f"""\nMAXJOBS {population} {max_jobs_running}\n\n"""

# Add parent-child relationship
file_content += f"""PARENT {' '.join(parent_list)} CHILD {combine_job}\n"""

# Write to file
with open(dag_filename, "w") as file:
    file.write(file_content)

print(f"DAG file written to {dag_filename}")
