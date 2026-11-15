FROM dpc-tests-libero-r3m:latest AS system
FROM dpc-nldl-paper-buildout:latest

COPY --from=system /opt/conda/share/zoneinfo /usr/share/zoneinfo
COPY robosuite /opt/robosuite
COPY robomimic /opt/robomimic
COPY mimicgen /opt/mimicgen
COPY robosuite-task-zoo /opt/robosuite-task-zoo
RUN site=/home/chengine/miniconda3/envs/darp_native/lib/python3.11/site-packages && \
    rm -rf "$site/robosuite" "$site/robomimic" "$site/mimicgen" "$site/robosuite_task_zoo" && \
    ln -s /opt/robosuite/robosuite "$site/robosuite" && \
    ln -s /opt/robomimic/robomimic "$site/robomimic" && \
    ln -s /opt/mimicgen/mimicgen "$site/mimicgen" && \
    ln -s /opt/robosuite-task-zoo/robosuite_task_zoo "$site/robosuite_task_zoo" && \
    chmod -R a+w /opt/mimicgen/mimicgen/models/robosuite/assets/shapenet_core
ENV PYTHONTZPATH=/usr/share/zoneinfo
