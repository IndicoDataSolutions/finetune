pipeline {
  agent any
  stages {
    stage('Build Docker Image') {
      steps {
        sh 'docker container rm -f finetune'
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
        sh 'docker exec finetune nosetests -sv --nologcapture'
        sh 'pwd && ls'
      }
    }
    stage('Remove container') {
      steps {
        sh 'docker rm -f finetune'
      }
    }
  }
}