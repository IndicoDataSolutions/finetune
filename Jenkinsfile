pipeline {
  agent any
  stages {
    stage('Build Docker Image') {
      steps {
        sh 'docker container rm -f finetune || true'
        sh './docker/build_docker.sh '
      }
    }
    stage('Start Docker Image') {
      steps {
        sh './docker/start_docker.sh'
      }
    }
    stage('Run Tests ') {
      steps {
        sh 'docker exec finetune nosetests'
        sh 'docker exec finetune nosetests -sv --nologcapture'
      }
    }
    stage('Remove container') {
      steps {
        sh 'docker rm -f finetune'
      }
    }
  }
}