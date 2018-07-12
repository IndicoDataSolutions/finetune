pipeline {
  agent any
  stages {
    stage('Build Docker Image') {
      steps {
        sh 'docker container rm -f finetune || true'
        sh './docker/build_docker.sh '
        sh 'ls -lah'
      }
    }
    stage('Start Docker Image') {
      steps {
        sh 'docker run --runtime=nvidia -d -v $PWD:/Finetune --name finetune finetune '
        sh 'docker logs finetune'
      }
    }
    stage('Run Tests ') {
      steps {
        sh 'docker exec finetune nosetests'
      }
    }
    stage('Remove container') {
      steps {
        sh 'docker rm -f finetune'
      }
    }
  }
}